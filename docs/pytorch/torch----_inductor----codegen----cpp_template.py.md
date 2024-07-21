# `.\pytorch\torch\_inductor\codegen\cpp_template.py`

```py
# 引入必要的模块和库
# mypy: allow-untyped-defs 允许未类型化的定义（针对静态类型检查工具mypy的设置）
import ctypes  # 提供与C语言兼容的数据类型
import functools  # 提供高阶函数操作的工具
import itertools  # 提供用于创建迭代器的函数
import logging  # 提供日志记录功能

import sys  # 提供与Python解释器交互的变量和函数
from typing import Callable, List, Optional  # 提供静态类型检查所需的类型提示
from unittest.mock import patch  # 提供单元测试时的模拟对象功能

import sympy  # 提供符号数学计算的支持

# 引入项目内部模块和库
from .. import codecache, config, ir  # 导入项目内部的代码缓存、配置和中间表示模块
from ..autotune_process import CppBenchmarkRequest, TensorMeta  # 导入自动调优过程中使用的请求和张量元数据类
from ..utils import IndentedBuffer, Placeholder, unique  # 导入缩进缓冲区、占位符、唯一化函数等工具函数
from ..virtualized import V  # 导入虚拟化相关的模块和类
from .common import KernelTemplate  # 导入通用的核模板基类
from .cpp_template_kernel import CppTemplateCaller, CppTemplateKernel  # 导入C++模板调用器和C++模板核心类

# 设置日志记录器，使用当前模块的名称
log = logging.getLogger(__name__)

# 定义一个C++模板类，继承自KernelTemplate基类
class CppTemplate(KernelTemplate):
    # 用于生成索引的计数器，每次生成的索引递增
    index_counter = itertools.count()

    # 初始化方法，接受名称、输入节点、布局、线程数和尾声创建器作为参数
    def __init__(
        self,
        name: str,
        input_nodes,  # 输入节点列表
        layout: ir.Layout,  # 布局对象
        num_threads: int,  # 线程数
        epilogue_creator: Optional[Callable[[ir.Buffer], ir.Pointwise]] = None,  # 可选的尾声创建器函数
    ):
        super().__init__(name)  # 调用父类的初始化方法
        self.input_nodes = input_nodes  # 设置输入节点列表
        self.output_node: ir.Buffer = ir.Buffer("buf_out", layout)  # 设置输出节点，使用指定布局的缓冲区
        self.layout = layout  # 设置布局对象
        self.num_threads = num_threads  # 设置线程数
        self.epilogue_creator = epilogue_creator  # 设置尾声创建器函数（如果提供）
    # 定义一个生成函数，接受关键字参数 kwargs
    def generate(self, **kwargs):
        # 构建内核名称，使用当前对象的名称生成
        kernel_name = f"cpp_{self.name}"
        # 在 V.graph 上下文中，替换 get_dtype 方法为 self.output_node 的假方法 _fake_get_dtype
        with patch.object(
            V.graph, "get_dtype", self._fake_get_dtype(self.output_node)
        ), \
        # 同时替换 FlexibleLayout 类的 allow_indexing 属性为 True
        patch.object(ir.FlexibleLayout, "allow_indexing", True), \
        # 创建 CppTemplateKernel 对象，内核名称为 kernel_name，线程数为 self.num_threads
        CppTemplateKernel(
            kernel_name=kernel_name, num_threads=self.num_threads
        ) as kernel:
            # 使用 kernel 渲染代码模板，传入当前对象和关键字参数 kwargs
            code = kernel.render(self, **kwargs)
            # 获取调用参数的 Python 定义
            _, call_args, _, _ = kernel.args.python_argdefs()
            # 记录调试日志，输出生成的代码
            log.debug("Generated Code:\n%s", code)
            # 记录调试日志，输出参数信息，包括 cpp_argdefs 和 python_argdefs
            log.debug(
                "Args: cpp_argdefs: %s, python_argdefs: %s",
                kernel.args.cpp_argdefs(),
                kernel.args.python_argdefs(),
            )

        # 准备预期的参数列表，包括所有输入节点的名称和输出节点的名称
        expected_args = list(
            unique(input_node.get_name() for input_node in self.input_nodes)
        )
        expected_args.extend([self.output_node.get_name()])
        # 断言调用参数的前部分与预期的参数列表一致
        assert list(call_args)[: len(expected_args)] == expected_args, (
            call_args,
            expected_args,
        )
        # 获取额外参数，这些参数根据调用参数的剩余部分进行计算和转换
        extra_args = V.graph.sizevars.size_hints(
            map(sympy.expand, call_args[len(expected_args) :])
        )
        # 将 size hint 明确转换为 ctypes.c_ulonglong 类型，因为在 cpp 内核中，绑定到 C long
        extra_args = tuple(ctypes.c_ulonglong(x) for x in extra_args)

        # 构建内核哈希名称，包含 self.name 和当前索引计数器的下一个值
        kernel_hash_name = f"cpp_{self.name}_{next(self.index_counter)}"

        # 创建用于 CPP 的 BenchmarkRequest
        bmreq = CppBenchmarkRequest(
            kernel_name=kernel_name,
            input_tensor_meta=TensorMeta.from_irnodes(self.input_nodes),
            output_tensor_meta=TensorMeta.from_irnodes(self.output_node),
            extra_args=extra_args,
            source_code=code,
        )

        # 定义一个生成内核渲染的函数
        def make_kernel_render(
            template_node: ir.CppTemplateBuffer,
            epilogue_nodes: Optional[List[ir.IRNode]] = None,
        ):
            # 创建 CppTemplateKernel 对象，内核名称为 Placeholder.KERNEL_NAME，线程数为 self.num_threads
            kernel = CppTemplateKernel(
                kernel_name=str(Placeholder.KERNEL_NAME), num_threads=self.num_threads
            )
            # 创建一个部分渲染函数，用于渲染当前对象、模板缓冲节点和尾声节点等参数
            render = functools.partial(
                kernel.render,
                self,
                template_buffer_node=template_node,
                epilogue_nodes=epilogue_nodes,
                **kwargs,
            )
            return kernel, render

        # 返回一个 CppTemplateCaller 对象，包含内核哈希名称、当前对象名称、输入节点、输出节点布局、make_kernel_render 函数和 bmreq 对象
        return CppTemplateCaller(
            kernel_hash_name,
            self.name,
            self.input_nodes,
            self.output_node.get_layout(),
            make_kernel_render,
            bmreq,
            self,
        )
    # 定义一个方法 `header`，返回一个 `IndentedBuffer` 对象
    def header(self) -> IndentedBuffer:
        # 创建一个空的 `IndentedBuffer` 对象
        res = IndentedBuffer()
        # 调用 `codecache.cpp_prefix()` 方法，并将结果写入 `res` 对象
        res.writeline(codecache.cpp_prefix())
        # 将字符串 `#include "c10/util/Unroll.h"` 添加到 `res` 对象中
        res.splice(
            """
                #include "c10/util/Unroll.h"
            """
        )
        # 检查是否启用内核性能分析，并且系统平台是 Linux
        enable_kernel_profile = (
            config.cpp.enable_kernel_profile and sys.platform == "linux"
        )
        # 如果启用了内核性能分析，则将 `#include <ATen/record_function.h>` 添加到 `res` 对象中
        if enable_kernel_profile:
            res.writelines(["#include <ATen/record_function.h>"])
        # 返回填充后的 `IndentedBuffer` 对象 `res`
        return res

    # 定义一个抽象方法 `render`，暂时未实现
    def render(self, **kwargs) -> str:
        # 抛出未实现错误，提示子类需要实现这个方法
        raise NotImplementedError
```