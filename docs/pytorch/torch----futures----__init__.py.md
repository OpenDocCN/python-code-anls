# `.\pytorch\torch\futures\__init__.py`

```py
# mypy: allow-untyped-defs
# 导入 __future__ 模块中的 annotations 功能
from __future__ import annotations

# 导入所需的类型提示和类型定义
from typing import cast, Callable, Generic, List, Optional, Type, TypeVar, Union

# 导入 PyTorch 库
import torch

# 定义模块的公开接口
__all__ = ['Future', 'collect_all', 'wait_all']

# 定义类型变量 T 和 S
T = TypeVar("T")
S = TypeVar("S")

# 定义元类 _PyFutureMeta，继承自 torch._C.Future 类和 Generic 类
class _PyFutureMeta(type(torch._C.Future), type(Generic)):  # type: ignore[misc, no-redef]
    pass

# 定义 Future 类，继承自 torch._C.Future 和泛型 Generic 类
class Future(torch._C.Future, Generic[T], metaclass=_PyFutureMeta):
    r"""
    封装了一个 ``torch._C.Future`` 对象，用于异步执行可调用对象，
    如 :meth:`~torch.distributed.rpc.rpc_async`。它还提供了一组API，
    用于添加回调函数和设置结果。

    .. warning:: GPU 支持是一个测试功能，可能会发生变化。
    """

    def __init__(self, *, devices: Optional[List[Union[int, str, torch.device]]] = None):
        r"""
        创建一个空的未设置的 ``Future`` 对象。如果未来的值包含 CUDA 张量，
        必须在构造时指定它们的 CUDA 设备（如果 ``torch.cuda.is_available()`` 返回
        ``True``）。这是为了确保适当的 CUDA 流同步。子 Future 对象，由 ``then`` 方法返回，
        将继承这些设备。

        Args:
            devices(``List[Union[int, str, torch.device]]``, optional): 未来值中允许存在的张量的设备集合，
                以及允许回调操作的设备。
        """
        if devices is None:
            devices = []
        # 调用父类初始化方法，将设备列表转换为 torch.device 对象的列表
        super().__init__([torch.device(d) for d in devices])

    def done(self) -> bool:
        r"""
        返回 ``True`` 如果该 ``Future`` 已完成。一个 ``Future`` 在具有结果或异常时被认为已完成。

        如果值包含驻留在 GPU 上的张量，则即使异步内核尚未完成在设备上运行，
        ``Future.done()`` 也会返回 ``True``，因为在这个阶段结果已经可用，只要执行适当的同步操作（参见 :meth:`wait`）。
        """
        # 调用父类的 done() 方法，判断该 Future 是否已完成
        return super().done()
    # 等待此 Future 对象的值就绪并返回该值

    def wait(self) -> T:
        r"""
        阻塞直到此 ``Future`` 对象的值就绪。

        如果值包含驻留在 GPU 上的张量，则会执行额外的同步操作，与可能异步填充这些张量的内核（在设备上执行）同步。
        这种同步是非阻塞的，这意味着 ``wait()`` 会在当前流中插入必要的指令，以确保后续在这些流上排队的操作将在异步内核之后正确调度，
        但是一旦完成这些操作，即使这些内核仍在运行，``wait()`` 也会返回。
        访问和使用值时不需要进一步的同步，只要不改变流。

        Returns:
            此 ``Future`` 对象持有的值。如果创建值的函数（回调或 RPC）抛出错误，此 ``wait`` 方法也会抛出错误。
        """
        return super().wait()

    # 获取已完成 Future 对象的值

    def value(self) -> T:
        r"""
        获取已经完成的 Future 对象的值。

        此方法应在调用 :meth:`wait` 后调用，或者在传递给 :meth:`then` 的回调函数内部调用。
        在其他情况下，此 ``Future`` 可能尚未持有值，并且调用 ``value()`` 可能会失败。

        如果值包含驻留在 GPU 上的张量，则此方法将 *不会* 执行任何额外的同步。
        这应该在之前通过调用 :meth:`wait` 分开进行，除非在回调中（此时 :meth:`then` 已经处理）。

        Returns:
            此 ``Future`` 对象持有的值。如果创建值的函数（回调或 RPC）抛出错误，此 ``value()`` 方法也会抛出错误。
        """
        return super().value()
    def add_done_callback(self, callback: Callable[[Future[T]], None]) -> None:
        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed. The callback must take one argument, which is the
        reference to this ``Future``. The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run inline.

        We recommend that you use the :meth:`then` method as it provides a way
        to synchronize after your callback has completed. ``add_done_callback``
        can be cheaper if your callback does not return anything. But both
        :meth:`then` and ``add_done_callback`` use the same callback
        registration API under the hood.

        With respect to GPU tensors, this method behaves in the same way as
        :meth:`then`.

        Args:
            callback(``Future``): a ``Callable`` that takes in one argument,
                which is the reference to this ``Future``.

        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback,
            error handling must be carefully taken care of. For example, if
            this callback later completes additional futures, those futures are
            not marked as completed with an error and the user is responsible
            for handling completion/waiting on those futures independently.

        Example::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print("This will run after the future has finished.")
            ...     print(fut.wait())
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            This will run after the future has finished.
            5
        """
        super().add_done_callback(callback)



        r"""
        Append the given callback function to this ``Future``, which will be run
        when the ``Future`` is completed.  Multiple callbacks can be added to
        the same ``Future``, but the order in which they will be executed cannot
        be guaranteed. The callback must take one argument, which is the
        reference to this ``Future``.
        """
        # 将给定的回调函数添加到此“Future”中，当“Future”完成时将运行它
        # 可以向同一个“Future”中添加多个回调函数，但无法保证它们的执行顺序
        # 回调函数必须接受一个参数，即对此“Future”的引用

        r"""
        The callback function can use the
        :meth:`value` method to get the value. Note that if this ``Future`` is
        already completed, the given callback will be run inline.
        """
        # 回调函数可以使用 :meth:`value` 方法获取值
        # 如果此“Future”已经完成，给定的回调将会直接运行

        r"""
        We recommend that you use the :meth:`then` method as it provides a way
        to synchronize after your callback has completed. ``add_done_callback``
        can be cheaper if your callback does not return anything. But both
        :meth:`then` and ``add_done_callback`` use the same callback
        registration API under the hood.
        """
        # 我们建议您使用 :meth:`then` 方法，因为它提供了在回调完成后进行同步的方法
        # 如果您的回调函数不返回任何内容，使用 ``add_done_callback`` 可能更便宜
        # 但是 :meth:`then` 和 ``add_done_callback`` 在底层使用相同的回调注册 API

        r"""
        With respect to GPU tensors, this method behaves in the same way as
        :meth:`then`.
        """
        # 关于 GPU 张量，此方法的行为与 :meth:`then` 相同

        r"""
        .. note:: Note that if the callback function throws, either
            through the original future being completed with an exception and
            calling ``fut.wait()``, or through other code in the callback,
            error handling must be carefully taken care of. For example, if
            this callback later completes additional futures, those futures are
            not marked as completed with an error and the user is responsible
            for handling completion/waiting on those futures independently.
        """
        # 注意：如果回调函数抛出异常，无论是通过原始的未来对象完成异常并调用 ``fut.wait()``，
        # 还是通过回调中的其他代码抛出异常，都必须仔细处理错误。例如，如果此回调稍后完成了其他未来对象，
        # 那些未来对象不会因为错误而被标记为完成，用户需要独立处理这些未来对象的完成/等待

        r"""
        Example::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> def callback(fut):
            ...     print("This will run after the future has finished.")
            ...     print(fut.wait())
            >>> fut = torch.futures.Future()
            >>> fut.add_done_callback(callback)
            >>> fut.set_result(5)
            This will run after the future has finished.
            5
        """
        # 示例：
        # >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
        # >>> def callback(fut):
        # ...     print("This will run after the future has finished.")
        # ...     print(fut.wait())
        # >>> fut = torch.futures.Future()
        # >>> fut.add_done_callback(callback)
        # >>> fut.set_result(5)
        # 这将在未来完成后运行
        # 5

        # 调用父类方法将回调函数添加到“Future”中
        super().add_done_callback(callback)
    def set_result(self, result: T) -> None:
        r"""
        Set the result for this ``Future``, which will mark this ``Future`` as
        completed and trigger all attached callbacks. Note that a ``Future``
        cannot be marked completed twice.

        If the result contains tensors that reside on GPUs, this method can be
        called even if the asynchronous kernels that are populating those
        tensors haven't yet completed running on the device, provided that the
        streams on which those kernels were enqueued are set as the current ones
        when this method is called. Put simply, it's safe to call this method
        immediately after launching those kernels, without any additional
        synchronization, as long as one doesn't change streams in between. This
        method will record events on all the relevant current streams and will
        use them to ensure proper scheduling for all the consumers of this
        ``Future``.

        Args:
            result (object): the result object of this ``Future``.

        Example::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> import threading
            >>> import time
            >>> def slow_set_future(fut, value):
            ...     time.sleep(0.5)
            ...     fut.set_result(value)
            >>> fut = torch.futures.Future()
            >>> t = threading.Thread(
            ...     target=slow_set_future,
            ...     args=(fut, torch.ones(2) * 3)
            ... )
            >>> t.start()
            >>> print(fut.wait())
            tensor([3., 3.])
            >>> t.join()
        """
        # 调用父类方法设置 Future 的结果
        super().set_result(result)

    def set_exception(self, result: T) -> None:
        r"""
        Set an exception for this ``Future``, which will mark this ``Future`` as
        completed with an error and trigger all attached callbacks. Note that
        when calling wait()/value() on this ``Future``, the exception set here
        will be raised inline.

        Args:
            result (BaseException): the exception for this ``Future``.

        Example::
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
            >>> fut = torch.futures.Future()
            >>> fut.set_exception(ValueError("foo"))
            >>> fut.wait()
            Traceback (most recent call last):
            ...
            ValueError: foo
        """
        # 断言 result 是异常对象，而不是普通对象
        assert isinstance(result, Exception), f"{result} is of type {type(result)}, not an Exception."

        # 定义一个函数来抛出异常
        def raise_error(fut_result):
            raise fut_result

        # 调用父类方法设置解封装函数，用于处理异常
        super()._set_unwrap_func(raise_error)
        # 调用 set_result 方法设置异常对象，实际上标记 Future 为完成，并触发所有已附加的回调
        self.set_result(result)  # type: ignore[arg-type]
# 将提供的 :class:`~torch.futures.Future` 对象收集到一个单独的组合 :class:`~torch.futures.Future` 中，
# 当所有子 Future 完成时，该组合 Future 也将完成。
def collect_all(futures: List[Future]) -> Future[List[Future]]:
    r"""
    Collects the provided :class:`~torch.futures.Future` objects into a single
    combined :class:`~torch.futures.Future` that is completed when all of the
    sub-futures are completed.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` objects.

    Returns:
        Returns a :class:`~torch.futures.Future` object to a list of the passed
        in Futures.

    Example::
        >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_FUTURES)
        >>> fut0 = torch.futures.Future()
        >>> fut1 = torch.futures.Future()
        >>> fut = torch.futures.collect_all([fut0, fut1])
        >>> fut0.set_result(0)
        >>> fut1.set_result(1)
        >>> fut_list = fut.wait()
        >>> print(f"fut0 result = {fut_list[0].wait()}")
        fut0 result = 0
        >>> print(f"fut1 result = {fut_list[1].wait()}")
        fut1 result = 1
    """
    return cast(Future[List[Future]], torch._C._collect_all(cast(List[torch._C.Future], futures)))


# 等待所有提供的 futures 完成，并返回已完成值的列表。
# 如果任何一个 future 遇到错误，方法将提前退出并报告错误，不会等待其他 futures 完成。
def wait_all(futures: List[Future]) -> List:
    r"""
    Waits for all provided futures to be complete, and returns
    the list of completed values. If any of the futures encounters an error,
    the method will exit early and report the error not waiting for other
    futures to complete.

    Args:
        futures (list): a list of :class:`~torch.futures.Future` object.

    Returns:
        A list of the completed :class:`~torch.futures.Future` results. This
        method will throw an error if ``wait`` on any
        :class:`~torch.futures.Future` throws.
    """
    # 使用列表推导式等待所有 futures 完成，并返回结果列表
    return [fut.wait() for fut in torch._C._collect_all(cast(List[torch._C.Future], futures)).wait()]
```