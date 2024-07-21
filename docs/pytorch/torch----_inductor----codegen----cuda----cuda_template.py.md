# `.\pytorch\torch\_inductor\codegen\cuda\cuda_template.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和类
import functools  # functools 模块用于高阶函数操作
import itertools  # itertools 模块提供了用于创建和操作迭代器的函数
import logging  # logging 模块用于记录日志信息
from typing import List, Optional  # 引入类型提示 List 和 Optional
from unittest.mock import patch  # 导入 patch 函数用于模拟对象

import sympy  # sympy 是一个用于符号数学的 Python 库

import torch  # 引入 PyTorch 模块
from ...autotune_process import CUDABenchmarkRequest, TensorMeta  # 导入相关类
from ...ir import Buffer, CUDATemplateBuffer, IRNode, Layout  # 导入缓冲区、CUDA 模板缓冲区、IR 节点和布局类

from ...utils import IndentedBuffer, unique  # 导入缩进缓冲区和唯一性函数
from ...virtualized import V  # 导入虚拟化模块 V
from ..common import KernelTemplate  # 导入通用内核模板类
from .cuda_kernel import CUDATemplateCaller, CUDATemplateKernel  # 导入 CUDA 内核调用和 CUDA 内核模板

log = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class CUDATemplate(KernelTemplate):
    index_counter = itertools.count()  # 创建一个迭代计数器

    def __init__(
        self,
        name: str,
        input_nodes: List[Buffer],
        layout: Layout,
        input_reorder: Optional[List[int]] = None,
    ):
        """
        Baseclass for CUDA C++ Templates, derived from KernelTemplate. Not to be instantiated directly.

        Args:
            name (str): The name of the CUDATemplate object.
            input_nodes (List[IRNode]): A list of input IRNodes.
            layout (Layout): The layout of the output buffer / tensor.
            input_reorder (Optional[List[int]]): An optional list that specifies the order of the input nodes.
        """
        super().__init__(name)  # 调用父类的构造方法
        self.input_nodes = input_nodes  # 初始化输入节点列表
        self.output_node: Buffer = Buffer("buf_out", layout)  # 创建输出缓冲区节点
        self.input_reorder = input_reorder  # 初始化输入重排序列表
        self.layout = layout  # 设置布局信息

    def generate(  # type: ignore[override]
        self,
        **kwargs,
    ) -> CUDATemplateCaller:
        """
        Generates the CUDA template caller object for the given GEMM template and operation. This CUDATemplateCaller
        may be used to call and benchmark the generated CUDA kernel in a standalone manner to enable Autotuning.

        Args:
            kwargs: Additional keyword arguments.

        Returns:
            A CUDATemplateCaller object representing the generated CUDA template caller.
        """
        kernel_name = f"cuda_{self.name}"
        # 使用类的名称生成 CUDA 内核的名称

        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), CUDATemplateKernel(
            kernel_name=kernel_name,
        ) as kernel:
            # 使用模拟的数据类型和生成的 CUDA 内核对象，进入代码块

            code = self.render(kernel=kernel, **kwargs)
            # 生成 CUDA 内核代码，并传入 kernel 对象和额外的关键字参数 kwargs

            _, call_args, _, _ = kernel.args.python_argdefs()
            # 解析内核对象的 Python 参数定义

            log.debug("Generated Code:\n%s", code)
            # 记录生成的 CUDA 内核代码

            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )
            # 记录内核的 C++ 和 Python 参数定义

        input_reorder = (
            self.input_reorder
            if self.input_reorder is not None
            else list(range(len(self.input_nodes)))
        )
        # 如果输入重排存在，则使用它；否则创建一个索引列表

        expected_args = list(
            unique(self.input_nodes[idx].get_name() for idx in input_reorder)
        )
        expected_args.extend([self.output_node.get_name()])
        # 生成预期的参数列表，包括输入节点和输出节点的名称

        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        # 断言调用参数列表与预期参数列表相匹配

        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # 根据剩余的参数推断额外的参数信息

        kernel_hash_name = f"cuda_{self.name}_{next(self.index_counter)}"
        # 使用类的名称和索引计数器生成 CUDA 内核的哈希名称

        # 创建 BenchmarkRequest 对象
        bmreq = CUDABenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        def make_kernel_render(
            template_node: CUDATemplateBuffer,
            epilogue_nodes: Optional[List[IRNode]] = None,
        ):
            # 定义生成内核和渲染函数的方法
            kernel = CUDATemplateKernel(
                kernel_name="KERNEL_NAME",
            )
            render = functools.partial(
                self.render,
                kernel=kernel,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,  # includes "op" argument in case of CUTLASSGemmTemplate
            )
            return kernel, render

        # 返回 CUDATemplateCaller 对象
        return CUDATemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
            kwargs,
        )
    # 定义一个方法 `header`，返回一个缩进的代码块对象 IndentedBuffer
    def header(self) -> IndentedBuffer:
        # 创建一个空的 IndentedBuffer 对象
        res = IndentedBuffer()
        # 向 IndentedBuffer 对象中插入以下代码片段，这些代码将包含在生成的 C++ 头文件中
        res.splice(
            """
            #include <exception>
            #include <iostream>
            #include <memory>
            #include <random>
            #include <vector>
            """
        )
        # 返回填充好的 IndentedBuffer 对象
        return res

    # 定义一个方法 `globals`，返回一个缩进的代码块对象 IndentedBuffer
    def globals(self) -> IndentedBuffer:
        # 创建一个空的 IndentedBuffer 对象
        res = IndentedBuffer()
        # 向 IndentedBuffer 对象中插入以下代码片段，这些代码定义了导出符号和类型别名
        res.splice(
            """
            // We compile all models with -fvisibility=hidden. Any symbols that need to be
            // exposed in the final shared library must be declared with PT_EXPORT to make
            // them visible.
            #ifdef __GNUC__ // Applies to any compiler with GNU extensions (clang and g++)
            #define PT_EXPORT __attribute__((__visibility__("default")))
            #else
            #ifdef _WIN32
            #define PT_EXPORT __declspec(dllexport)
            #else
            #define PT_EXPORT
            #endif
            #endif
            using bfloat16 = nv_bfloat16;
            """
        )
        # 返回填充好的 IndentedBuffer 对象
        return res

    # 定义一个方法 `render`，该方法在当前实现中抛出 NotImplementedError
    def render(self, **kwargs) -> str:
        raise NotImplementedError
class CUTLASSTemplate(CUDATemplate):
    """
    CUTLASSTemplate is a class that provides a template for generating CUTLASS Templates. Used as a baseclass for the
    CUTLASSGemmTemplate, providing functionality that might also be relevant for non-GEMM CUTLASS Kernels.
    """

    def header(self) -> IndentedBuffer:
        res = super().header()
        res.splice(
            """
            #include "cute/tensor.hpp"
            #include "cutlass/cutlass.h"
            #include "cutlass/numeric_types.h"
            #include "cutlass/tensor_ref.h"
            #include "cutlass/util/host_tensor.h"
            #include "cutlass/util/reference/host/tensor_fill.h"
            #include "cutlass/util/reference/device/tensor_fill.h"
            #include "cutlass/util/device_memory.h"
            """
        )
        return res

    def globals(self) -> IndentedBuffer:
        res = super().globals()
        res.splice(
            """
            using namespace cute;
            #define CUTLASS_CHECK(status)                                                      \\
            {                                                                                  \\
              cutlass::Status error = status;                                                  \\
              if (error != cutlass::Status::kSuccess) {                                        \\
                auto msg = std::string("[") + __FILE__ + "] Got cutlass error: " +             \\
                    cutlassGetStatusString(error) + " at: " + std::to_string(__LINE__);        \\
                throw std::runtime_error(msg);                                                 \\
              }                                                                                \\
            }

            // Used as pass-through functor in EVT just for type casting / rounding
            template <typename T>
            struct identity_op {
              CUTLASS_HOST_DEVICE
              T operator()(T val) const { return val; }
            };
            """
        )
        return res

    def cute_int(self, int_str: str, var_name: str) -> str:
        res = ""
        if int_str in {"1", "1L"}:
            res = "cute::Int<1>{}"
        else:
            res = int_str

        return f"{res} /* {var_name} */"

    _DTYPE_TO_CUTLASS = {
        torch.float32: "float",
        torch.float64: "double",
        torch.float16: "cutlass::half_t",
        torch.int32: "int",
        torch.int8: "int8_t",
        torch.uint8: "uint8_t",
        torch.bool: "bool",
        torch.bfloat16: "cutlass::bfloat16_t",
    }

    def cutlass_type_cast(self, node: IRNode, ptr: str) -> str:
        if node is None:
            return ptr
        else:
            return f"({self._DTYPE_TO_CUTLASS.get(node.get_dtype())}*)({ptr})"
```