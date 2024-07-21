# `.\pytorch\torch\random.py`

```py
# mypy: allow-untyped-defs
# 引入上下文管理模块
import contextlib
# 引入警告模块
import warnings
# 引入生成器类型
from typing import Generator

# 引入PyTorch库
import torch
# 引入默认生成器
from torch._C import default_generator


def set_rng_state(new_state: torch.Tensor) -> None:
    r"""设置随机数生成器的状态。

    .. note:: 此函数仅适用于CPU。对于CUDA，请使用
        :func:`torch.manual_seed`，该函数适用于CPU和CUDA。

    Args:
        new_state (torch.ByteTensor): 所需的状态
    """
    # 调用默认生成器的设置状态方法
    default_generator.set_state(new_state)


def get_rng_state() -> torch.Tensor:
    r"""以`torch.ByteTensor`的形式返回随机数生成器的状态。

    .. note:: 返回的状态仅适用于CPU上的默认生成器。

    See also: :func:`torch.random.fork_rng`.
    """
    # 返回默认生成器的当前状态
    return default_generator.get_state()


def manual_seed(seed) -> torch._C.Generator:
    r"""设置在所有设备上生成随机数的种子。返回一个`torch.Generator`对象。

    Args:
        seed (int): 所需的种子。值必须在包含范围内，
            `[-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]`。否则，将引发运行时错误。
            负输入将使用公式`0xffff_ffff_ffff_ffff + seed`重新映射为正值。
    """
    # 将种子转换为整数
    seed = int(seed)
    # 导入CUDA模块
    import torch.cuda
    # 如果不处于错误分支中，设置所有CUDA设备的种子
    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    # 导入MPS模块
    import torch.mps
    # 如果不处于错误分支中，设置MPS设备的种子
    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    # 导入XPU模块
    import torch.xpu
    # 如果不处于错误分支中，设置所有XPU设备的种子
    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    # 调用私有设备的种子方法
    _seed_custom_device(seed)

    # 返回默认生成器的手动设置种子结果
    return default_generator.manual_seed(seed)


def seed() -> int:
    r"""将生成随机数的种子设置为所有设备上的非确定性
    随机数。返回一个用于种子RNG的64位数。
    """
    # 获取默认生成器的种子
    seed = default_generator.seed()
    # 导入CUDA模块
    import torch.cuda
    # 如果不处于错误分支中，设置所有CUDA设备的种子
    if not torch.cuda._is_in_bad_fork():
        torch.cuda.manual_seed_all(seed)

    # 导入MPS模块
    import torch.mps
    # 如果不处于错误分支中，设置MPS设备的种子
    if not torch.mps._is_in_bad_fork():
        torch.mps.manual_seed(seed)

    # 导入XPU模块
    import torch.xpu
    # 如果不处于错误分支中，设置所有XPU设备的种子
    if not torch.xpu._is_in_bad_fork():
        torch.xpu.manual_seed_all(seed)

    # 调用私有设备的种子方法
    _seed_custom_device(seed)

    # 返回生成的种子
    return seed


def _seed_custom_device(seed) -> None:
    r"""设置用于自定义设备生成随机数的种子。

    Args:
        seed (int): 所需的种子。

    See [Note: support the custom device with privateuse1]
    """
    # 将种子转换为整数
    seed = int(seed)
    # 获取私有设备1的后端名称
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    # 检查是否存在自定义的后端名称，例如在 torch 模块中
    if hasattr(torch, custom_backend_name):
        # 获取指定名称的自定义设备模块对象
        custom_device_mod = getattr(torch, custom_backend_name)
        # 定义用于检查坏分支的属性名称
        _bad_fork_name = "_is_in_bad_fork"
        # 定义手动设置所有种子的属性名称
        _seed_all_name = "manual_seed_all"
        
        # 如果自定义设备模块中同时存在坏分支检查和手动设置所有种子的方法
        if hasattr(custom_device_mod, _bad_fork_name) and hasattr(
            custom_device_mod, _seed_all_name
        ):
            # 如果当前未处于坏分支状态，则调用手动设置所有种子方法设置种子值
            if not getattr(custom_device_mod, _bad_fork_name)():
                getattr(custom_device_mod, _seed_all_name)(seed)
        else:
            # 构建警告信息，说明设置种子对于指定的自定义后端设备无效，提示用户添加相关 API 方法
            message = f"Set seed for `{custom_backend_name}` device does not take effect, please add API's "
            message += f"`{_bad_fork_name}` and `{_seed_all_name}` to `{custom_backend_name}` device module."
            # 发出用户警告，指出设置种子无效
            warnings.warn(message, UserWarning, stacklevel=3)
`
def initial_seed() -> int:
    r"""Returns the initial seed for generating random numbers as a
    Python `long`.

    .. note:: The returned seed is for the default generator on CPU only.
    """
    # 返回生成随机数的初始种子，使用默认生成器
    return default_generator.initial_seed()


_fork_rng_warned_already = False  # 初始化一个全局变量，标记警告是否已经发出

@contextlib.contextmanager
def fork_rng(
    devices=None,
    enabled=True,
    _caller="fork_rng",
    _devices_kw="devices",
    device_type="cuda",
) -> Generator:
    """
    Forks the RNG, so that when you return, the RNG is reset
    to the state that it was previously in.

    Args:
        devices (iterable of Device IDs): devices for which to fork
            the RNG. CPU RNG state is always forked. By default, :meth:`fork_rng` operates
            on all devices, but will emit a warning if your machine has a lot
            of devices, since this function will run very slowly in that case.
            If you explicitly specify devices, this warning will be suppressed
        enabled (bool): if ``False``, the RNG is not forked.  This is a convenience
            argument for easily disabling the context manager without having
            to delete it and unindent your Python code under it.
        device_type (str): device type str, default is `cuda`. As for custom device,
            see details in [Note: support the custom device with privateuse1]
    """

    # 将 device_type 转换为 torch.device 类型，并获取其类型字符串
    device_type = torch.device(device_type).type
    # 获取对应的 torch 模块
    device_mod = getattr(torch, device_type, None)
    # 如果未找到对应的 torch 模块，抛出运行时错误
    if device_mod is None:
        raise RuntimeError(
            f"torch has no module of `{device_type}`, you should register "
            + "a module by `torch._register_device_module`."
        )
    global _fork_rng_warned_already

    # 内部参数：
    #   _caller: 调用 fork_rng 的函数
    #   _devices_kw: _caller 中 devices 的关键字

    if not enabled:
        yield  # 如果未启用，直接返回
        return

    # 这里可以添加更多代码逻辑来处理 RNG 的分叉和重置
    # 如果未指定设备列表，则获取系统上可用设备的数量
    if devices is None:
        # 查询当前设备模块上的设备数量
        num_devices = device_mod.device_count()
        # 如果检测到多于一个设备，并且之前未发出警告
        if num_devices > 1 and not _fork_rng_warned_already:
            # 构造警告信息，提醒用户可能需要显式指定设备以避免性能问题
            message = (
                f"{device_type.upper()} reports that you have {num_devices} available devices, and "
                f"you have used {_caller} without explicitly specifying which devices are being used. "
                f"For safety, we initialize *every* {device_type.upper()} device by default, which can "
                f"be quite slow if you have a lot of {device_type.upper()}s. If you know that you are only"
                f" making use of a few {device_type.upper()} devices, set the environment variable "
                f"{device_type.upper()}_VISIBLE_DEVICES or the '{_devices_kw}' keyword argument of {_caller} "
                "with the set of devices you are actually using. For example, if you are using CPU only, "
                "set device.upper()_VISIBLE_DEVICES= or devices=[]; if you are using device 0 only, "
                f"set {device_type.upper()}_VISIBLE_DEVICES=0 or devices=[0].  To initialize all devices "
                f"and suppress this warning, set the '{_devices_kw}' keyword argument to "
                f"`range(torch.{device_type}.device_count())`."
            )
            # 发出警告信息
            warnings.warn(message)
            # 标记已经发出警告，避免重复发出
            _fork_rng_warned_already = True
        # 将设备列表设置为包含所有设备编号的列表
        devices = list(range(num_devices))
    else:
        # 如果用户传入了一个生成器，需要转换为列表以便多次遍历
        devices = list(devices)

    # 保存当前CPU的随机数生成器状态
    cpu_rng_state = torch.get_rng_state()
    # 获取每个设备的随机数生成器状态并保存在列表中
    device_rng_states = [device_mod.get_rng_state(device) for device in devices]

    try:
        # 执行被装饰函数的主体代码
        yield
    finally:
        # 在最后确保恢复CPU的随机数生成器状态
        torch.set_rng_state(cpu_rng_state)
        # 逐个将设备的随机数生成器状态恢复到原来的状态
        for device, device_rng_state in zip(devices, device_rng_states):
            device_mod.set_rng_state(device_rng_state, device)
```