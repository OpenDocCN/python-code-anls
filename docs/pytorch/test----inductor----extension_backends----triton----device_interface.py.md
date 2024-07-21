# `.\pytorch\test\inductor\extension_backends\triton\device_interface.py`

```py
from __future__ import annotations

import time  # 导入时间模块

from torch._dynamo import device_interface  # noqa: PLC2701 import-private-name


class DeviceProperties:
    def __init__(self) -> None:
        self.major = 8  # 设备主版本号，默认为8
        self.max_threads_per_multi_processor = 1  # 每个多处理器的最大线程数，默认为1
        self.multi_processor_count = 80  # 多处理器的数量，默认为80


class DeviceInterface(device_interface.DeviceInterface):
    class Event(
        device_interface._EventBase
    ):  # pyright: ignore [reportPrivateImportUsage]
        def __init__(
            self,
            enable_timing: bool = False,
            blocking: bool = False,
            interprocess: bool = False,
        ) -> None:
            self.enable_timing = enable_timing  # 是否启用计时，默认为False
            self.recorded_time: int | None = None  # 记录事件的时间戳，默认为None

        def record(self, stream) -> None:
            if not self.enable_timing:
                return  # 如果未启用计时，则不记录时间

            assert self.recorded_time is None  # 确保当前时间戳为空
            self.recorded_time = time.perf_counter_ns()  # 记录当前的性能计数器时间戳

        def elapsed_time(self, end_event: DeviceInterface.Event) -> float:
            assert self.recorded_time  # 确保已记录时间戳
            assert end_event.recorded_time  # 确保结束事件也有记录的时间戳
            # 计算经过的时间，单位为毫秒
            return (end_event.recorded_time - self.recorded_time) / 1000000

        def wait(self, stream) -> None:
            pass  # 等待事件完成的方法，这里为空实现

        def query(self) -> None:
            pass  # 查询事件状态的方法，这里为空实现

        def synchronize(self) -> None:
            pass  # 同步事件的方法，这里为空实现

    class device:  # noqa: N801 invalid-class-name # pyright: ignore [reportIncompatibleVariableOverride]
        def __init__(self, device) -> None:
            self.device = device  # 初始化设备属性

    class Worker(device_interface.DeviceInterface.Worker):
        @staticmethod
        def set_device(device: int) -> None:
            # 我们的后端没有设备索引
            pass  # 设置设备的静态方法，这里为空实现

        @staticmethod
        def current_device() -> int:
            # 我们的后端没有设备索引
            return 0  # 当前设备索引始终为0

        @staticmethod
        def get_device_properties(
            device=None,
        ) -> DeviceProperties:
            return DeviceProperties()  # 获取设备属性的静态方法，返回DeviceProperties实例

    @staticmethod
    def current_device() -> int:
        return 0  # 当前设备索引始终为0的静态方法

    @staticmethod
    def set_device(device) -> None:
        pass  # 设置设备的静态方法，这里为空实现

    @staticmethod
    def device_count() -> int:
        raise NotImplementedError  # 设备数量的静态方法，未实现

    @staticmethod
    def maybe_exchange_device(device: int) -> int:
        assert (
            device == 0
        ), f"Only device index 0 is supported, tried to set index to {device}"  # 断言设备索引为0
        return 0  # 前一个设备始终为0

    @staticmethod
    def exchange_device(device: int) -> int:
        assert (
            device == 0
        ), f"Only device index 0 is supported, tried to set index to {device}"  # 断言设备索引为0
        return 0  # 前一个设备始终为0

    @staticmethod
    def current_stream():
        raise NotImplementedError  # 当前流的静态方法，未实现

    @staticmethod
    def set_stream(stream) -> None:
        raise NotImplementedError  # 设置流的静态方法，未实现

    @staticmethod
    # 定义一个函数，根据设备索引获取原始数据流，返回空值
    def get_raw_stream(device_index: int):
        return None
    
    # 静态方法，用于同步设备
    @staticmethod
    def synchronize(device) -> None:
        pass
    
    # 静态方法，获取设备属性，返回设备属性对象
    @staticmethod
    def get_device_properties(device) -> DeviceProperties:
        raise NotImplementedError
    
    # 静态方法，检查设备是否可用，返回布尔值
    # 可以被 @patch 装饰器模拟替换
    @staticmethod
    def is_available() -> bool:
        return True
    
    # 静态方法，获取设备的计算能力，返回整数
    @staticmethod
    def get_compute_capability(device) -> int:
        return 0
```