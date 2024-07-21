# `.\pytorch\torch\_inductor\codegen\rocm\rocm_benchmark_request.py`

```
# mypy: allow-untyped-defs
from __future__ import annotations

import functools  # 导入 functools 模块，用于创建偏函数
import logging  # 导入 logging 模块，用于记录日志
from ctypes import byref, c_size_t, c_void_p  # 导入 ctypes 中的相关类和函数
from typing import Any, Callable, Iterable, List, Optional, Union  # 导入用于类型提示的模块

import torch  # 导入 PyTorch 库

from torch._inductor.autotune_process import GPUDeviceBenchmarkRequest, TensorMeta  # 导入自定义模块中的类和函数
from torch._inductor.codecache import DLLWrapper, ROCmCodeCache  # 导入自定义模块中的类和函数

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class ROCmBenchmarkRequest(GPUDeviceBenchmarkRequest):
    # Instances of this class need to be serializable across process boundaries.
    # Do not use CUDA Tensors here!

    def __init__(
        self,
        kernel_name: str,
        input_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        output_tensor_meta: Union[TensorMeta, List[TensorMeta]],
        extra_args: Iterable[Any],
        source_code: str,
    ):
        super().__init__(kernel_name, input_tensor_meta, output_tensor_meta, extra_args)
        self.source_code = source_code  # 初始化属性，存储传入的源代码
        self.workspace_size: int = 0  # 初始化属性，工作空间大小为零
        self.workspace: Optional[torch.Tensor] = None  # 初始化属性，工作空间张量为 None
        self.DLL: Optional[DLLWrapper] = None  # 初始化属性，DLL 包装器为 None
        self._workspace_size_updated = False  # 初始化属性，工作空间大小是否已更新标志为 False
        self.hash_key: str = ""  # 初始化属性，哈希键为空字符串
        self.source_file: str = ""  # 初始化属性，源文件路径为空字符串
        # 将源代码写入到 ROCmCodeCache 中，并获取哈希键和源文件路径
        self.hash_key, self.source_file = ROCmCodeCache.write(self.source_code, "so")

    def precompile(self):
        # 预编译函数，将源代码编译并缓存起来
        # 可能在单独的线程池中进行
        log.debug("Precompiling %s", self)  # 记录调试信息，表示正在预编译当前对象
        ROCmCodeCache.compile(self.source_code, "so")  # 调用 ROCmCodeCache 中的编译函数
        log.debug("Done precompiling %s", self)  # 记录调试信息，表示预编译完成

    def make_run_fn(
        self, *input_tensors: torch.Tensor, output_tensor: torch.Tensor
    ) -> Callable[[], None]:
        self.ensure_dll_loaded()  # 确保 DLL 已加载
        self.update_workspace_size()  # 更新工作空间大小
        args = [
            c_void_p(tensor.data_ptr())
            for tensor in list(input_tensors) + [output_tensor]
        ]  # 创建参数列表，包括输入张量和输出张量的指针

        log.debug(
            "make_run_fn: self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )  # 记录调试信息，输出函数运行所需的关键信息

        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)  # 获取当前 CUDA 流的指针
        run_method = getattr(self.DLL, self.kernel_name)  # 获取 DLL 对象中的指定核函数

        workspace_ptr = c_void_p(0)  # 初始化工作空间指针为零
        if self.workspace_size > 0:
            self.workspace = torch.zeros(
                (self.workspace_size + 7) // 8,
                dtype=torch.float64,
                device=output_tensor.device,
            )  # 创建工作空间张量，用于存储中间结果
            workspace_ptr = c_void_p(self.workspace.data_ptr())  # 获取工作空间张量的数据指针

        # 生成偏函数
        return functools.partial(
            run_method,
            *args,
            *self.extra_args,
            None,  # 空工作空间大小指针
            workspace_ptr,  # 设置工作空间指针
            stream_ptr,
        )  # 返回生成的偏函数对象
    def update_workspace_size(self) -> None:
        # 如果已经更新过 workspace_size，则直接返回，避免重复更新
        if self._workspace_size_updated:
            return
        # 确保 DLL 被加载
        self.ensure_dll_loaded()
        # 计算输入张量的唯一名称数目
        unique_input_count = len({meta.name for meta in self.input_tensor_meta})
        # 创建一个由 c_void_p 对象组成的列表，用于传递给 DLL 方法
        args = [c_void_p(None) for _ in range(unique_input_count + 1)]
        # 获取当前 CUDA 流的指针
        stream_ptr = c_void_p(torch.cuda.current_stream().cuda_stream)

        # 获取 DLL 中的运行方法（kernel_name 对应的方法）
        run_method = getattr(self.DLL, self.kernel_name)
        # 用于存储 workspace_size 的 c_size_t 对象
        c_workspace_size = c_size_t()
        # 调用 DLL 方法来获取 workspace_size，并初始化 workspace
        run_method(
            *args,  # 输入和输出指针
            *self.extra_args,  # 额外的参数
            byref(
                c_workspace_size
            ),  # 设置工作空间大小的指针，用于获取工作空间大小
            None,  # 空的工作空间指针
            stream_ptr,  # CUDA 流的指针
        )
        # 同步 CUDA，以便检查是否有 CUDA 错误发生
        torch.cuda.synchronize()
        # 将获取到的 workspace_size 赋值给 self.workspace_size
        self.workspace_size = c_workspace_size.value
        # 记录调试信息，包括更新后的 workspace_size 和其他相关变量
        log.debug(
            "update_workspace_size called: new workspace size=%d, self.kernel_name=%s, self.source_file=%s, self.hash_key=%s, self.DLL=%s, args=%s, self.extra_args=%s",  # noqa: B950
            self.workspace_size,
            self.kernel_name,
            self.source_file,
            self.hash_key,
            self.DLL,
            args,
            self.extra_args,
        )
        # 标记 workspace_size 已经更新
        self._workspace_size_updated = True

    def ensure_dll_loaded(self):
        # 如果 DLL 尚未加载，则加载对应的 DLL 和相关信息
        if self.DLL is None:
            self.DLL, self.hash_key, self.source_file = ROCmCodeCache.load(
                self.source_code, "so"
            )

    def cleanup_run_fn(self) -> None:
        # 如果 DLL 不为空，则关闭 DLL
        if self.DLL is not None:
            self.DLL.close()
        # 将 workspace 置为 None，清理运行函数
        self.workspace = None

    def __str__(self) -> str:
        # 返回对象的字符串表示，包括 kernel_name、source_file 和 hash_key
        return f"{self.kernel_name=}, {self.source_file=}, {self.hash_key=}"
```