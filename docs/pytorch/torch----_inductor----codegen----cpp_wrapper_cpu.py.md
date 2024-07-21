# `.\pytorch\torch\_inductor\codegen\cpp_wrapper_cpu.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import functools  # 提供函数式编程的工具
import math  # 提供数学函数
import os  # 提供与操作系统交互的功能
import sys  # 提供与Python解释器交互的功能
from itertools import count  # 提供无限计数器的迭代工具
from typing import Dict, List, Optional, Tuple  # 引入类型提示工具

import sympy  # 导入符号计算库Sympy
from sympy import Expr  # 导入Sympy表达式类

import torch  # 导入PyTorch深度学习框架

import torch._inductor.async_compile  # noqa: F401 required to warm up AsyncCompile pools
import torch._ops  # 导入PyTorch操作模块
from torch.fx.experimental.symbolic_shapes import ConvertIntKey, DivideByKey  # 导入符号形状相关工具
from .. import config, ir  # 导入相对路径的config和ir模块
from ..utils import _align, ALIGN_BYTES, cache_on_self, sympy_product  # 导入工具函数和常量
from ..virtualized import V  # 导入虚拟化相关模块
from .aoti_hipify_utils import maybe_hipify_code_wrapper  # 导入hipify相关工具函数
from .common import IndentedBuffer  # 导入缩进缓冲区工具类
from .cpp_utils import (
    cexpr,  # 导入C++表达式生成工具
    DEVICE_TO_ATEN,  # 设备到ATen张量的映射
    DTYPE_TO_ATEN,  # 数据类型到ATen张量的映射
    DTYPE_TO_CPP,  # 数据类型到C++类型的映射
    LAYOUT_TO_ATEN,  # 布局到ATen张量的映射
)
from .wrapper import EnterSubgraphLine, ExitSubgraphLine, WrapperCodeGen  # 导入包装器相关类

class CppWrapperCpu(WrapperCodeGen):
    """
    生成运行在CPU上的C++包装器并调用C++内核
    """

    def __init__(self):
        if not hasattr(self, "device"):
            self.device = "cpu"  # 如果对象没有device属性，则设为"cpu"
        super().__init__()  # 调用父类的构造方法
        self.declare = "auto "  # 自动类型推断声明关键字
        self.declare_maybe_reference = "decltype(auto) "  # 可能是引用类型的声明关键字
        self.ending = ";"  # 语句结束符号
        self.open_bracket = "{"  # 代码块开始符号
        self.closed_bracket = "}"  # 代码块结束符号
        self.comment = "//"  # 单行注释符号
        self.namespace = "at::"  # ATen命名空间
        self.none_str = "nullptr" if config.abi_compatible else "at::Tensor()"  # None的表示方式
        self.extern_call_ops = set()  # 外部调用操作的集合
        self.size = "sizes()"  # 张量尺寸的获取方法
        self.stride = "strides()"  # 张量步幅的获取方法
        self.cuda = False  # 是否使用CUDA加速
        self.supports_intermediate_hooks = False  # 是否支持中间挂钩
        self.outputs_need_copy = set()  # 需要复制的输出集合
        self.kernel_callsite_id = count()  # 内核调用位置的计数器
        self.var_array_id = count()  # 不同类型本地数组变量声明的计数器
        self.declared_var_array_vars = set()  # 已声明的本地数组变量集合
        self.int_array_id = count()  # int数组本地变量声明的计数器
        self.declared_int_array_vars = set()  # 已声明的int数组变量集合
        self.tmp_tensor_id = count()  # 临时张量本地变量声明的计数器
        self.arg_var_id = count()  # 参数变量声明的计数器
        self.used_cached_devices = set()  # 使用过的缓存设备集合
        self.used_cached_dtypes = set()  # 使用过的缓存数据类型集合
        self.used_cached_layouts = set()  # 使用过的缓存布局集合
        self.cached_output_id = count()  # 缓存输出标识的计数器
        self.scalar_to_tensor_id = count()  # 标量转换为张量的计数器
        self.custom_op_wrapper_loaded = False  # 是否加载了自定义操作包装器
        self.expr_printer = cexpr  # 表达式打印器设置为C++表达式生成工具

    def generate_kernel_call(
        self,
        name,
        call_args,
        grid=None,
        device_index=None,
        cuda=True,
        triton=True,
        arg_types=None,
        raw_args=None,
        grid_fn: str = "grid",
        triton_meta=None,
        # 生成调用C++内核的方法
        """
        生成内核调用代码。

        cuda: 定义后端是否为 GPU。否则后端为 CPU。

        triton: 定义 GPU 后端是否使用 Triton 进行代码生成。
                否则使用 CUDA 语言进行代码生成。
                仅当 cuda == True 时有效。
        """
        # 如果 cuda 为 True，则调用父类方法生成内核调用代码
        if cuda:
            return super().generate_kernel_call(
                name,
                call_args,
                grid,
                device_index,
                cuda,
                triton,
                arg_types,
                grid_fn,
            )
        else:
            # 如果 config.abi_compatible 为 True
            if config.abi_compatible:
                # 断言参数类型不为空，并且调用参数列表长度与参数类型列表长度相等
                assert arg_types is not None and len(call_args) == len(
                    arg_types
                ), "Mismatch call_args and arg_types in generate_kernel_call"
                new_args = []
                # 遍历调用参数列表
                for idx, arg in enumerate(call_args):
                    # 如果参数类型中包含 '*'，表示参数为指针类型
                    if "*" in arg_types[idx]:
                        var_name = f"var_{next(self.arg_var_id)}"
                        # 调用自身方法，生成获取数据指针的包装代码
                        self.writeline(
                            f"auto* {var_name} = get_data_ptr_wrapper({arg});"
                        )
                        new_args.append(f"({arg_types[idx]})({var_name})")
                    else:
                        # 参数为标量类型
                        new_args.append(arg)
                # 调用自身方法，包装内核调用并写入
                self.writeline(self.wrap_kernel_call(name, new_args))
            else:
                # 调用自身方法，包装内核调用并写入
                self.writeline(self.wrap_kernel_call(name, call_args))

    def write_constant(self, name, hashed):
        # 在头文件中写入带有哈希值的注释，以便我们的代码缓存为不同的常量生成不同的文件
        self.header.writeline(f"// {name} {hashed}")

    def mark_output_type(self):
        # 标记输出类型，以便将张量解包回 Python 标量
        from ..ir import ShapeAsConstantBuffer

        output_is_tensor = dict()
        # 遍历图的输出列表
        for idx, x in enumerate(V.graph.graph_outputs):
            if isinstance(x, ShapeAsConstantBuffer):
                output_is_tensor[idx] = False
            else:
                output_is_tensor[idx] = True

        self.output_is_tensor = output_is_tensor

    def write_prefix(self):
        # 如果是常量图，则不写入前缀，前缀将由主模块写入
        if V.graph.is_const_graph:
            return

        # 如果是 AOT 模式，则写入命名空间 torch 和 aot_inductor 的前缀
        if V.graph.aot_mode:
            self.prefix.writeline("namespace torch {")
            self.prefix.writeline("namespace aot_inductor {")

    def write_input_output_info(
        self,
        info_kind: str,
        idx: int,
        name: str,
    ):
        # 在前缀中写入输入/输出信息的格式化字符串
        self.prefix.writeline(f"""{info_kind}[{idx}].name = "{name}";""")
    # 获取输入的 C++ 类型表示
    def get_input_cpp_type(input):
        # 断言配置使用最小的 arrayref 接口
        assert config.use_minimal_arrayref_interface

        # 如果输入是 sympy.Expr 类型
        if isinstance(input, sympy.Expr):
            # 从图中导入 may_get_constant_buffer_dtype 函数
            from ..graph import may_get_constant_buffer_dtype

            # 获取 sympy.Expr 类型的常量缓冲区数据类型
            dtype = may_get_constant_buffer_dtype(input)
            # 断言成功获取到数据类型，否则报错
            assert dtype is not None, f"Failed to get the dtype of sympy.Expr: {input}"
            # 返回对应的 C++ 数据类型
            return DTYPE_TO_CPP[dtype]
        
        # 如果输入不是 sympy.Expr 类型，返回对应的 ArrayRefTensor 数据类型
        return f"ArrayRefTensor<{DTYPE_TO_CPP[input.get_dtype()]}>"

    # 生成输入张量的元素数量断言代码
    def codegen_input_numel_asserts(self):
        # 遍历图输入中的每个名称和缓冲区
        for name, buf in V.graph.graph_inputs.items():
            # 如果缓冲区是 sympy.Expr 类型，则跳过
            if isinstance(buf, sympy.Expr):
                continue

            # 比较张量大小为 0 的情况，目前忽略这些情况
            if sympy_product(buf.get_size()) == 0:
                continue
            
            # 获取张量的元素数量
            numel = buf.get_numel()
            # 向代码前缀写入断言语句
            self.prefix.writeline(f"assert_numel({name}, {numel});")

    # 生成张量数据类型变量声明的代码
    def codegen_tensor_dtype_var_decl(self, code: IndentedBuffer, name):
        # 如果配置是 ABI 兼容的
        if config.abi_compatible:
            # 声明 int32_t 类型的数据类型变量
            code.writeline(f"int32_t {name}_dtype;")
            # 调用 AOTI 接口获取张量的数据类型，并进行错误码检查
            code.writeline(
                "AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype"
                f"({name}, &{name}_dtype));"
            )
        else:
            # 如果不是 ABI 兼容的情况下，获取张量的数据类型
            code.writeline(f"auto {name}_dtype = {name}.dtype();")

    # 生成输入张量大小变量声明的代码
    def codegen_input_size_var_decl(self, code: IndentedBuffer, name):
        # 如果配置是 ABI 兼容的
        if config.abi_compatible:
            # 声明 int64_t 指针类型的大小变量
            code.writeline(f"int64_t* {name}_size;")
            # 调用 AOTI 接口获取张量的大小，并进行错误码检查
            code.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_sizes({name}, &{name}_size));"
            )
        else:
            # 如果不是 ABI 兼容的情况下，调用超类的方法获取输入张量的大小
            super().codegen_input_size_var_decl(code, name)

    # 生成输入张量步幅变量声明的代码
    def codegen_input_stride_var_decl(self, code: IndentedBuffer, name):
        # 如果配置是 ABI 兼容的
        if config.abi_compatible:
            # 声明 int64_t 指针类型的步幅变量
            code.writeline(f"int64_t* {name}_stride;")
            # 调用 AOTI 接口获取张量的步幅，并进行错误码检查
            code.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_strides({name}, &{name}_stride));"
            )
        else:
            # 如果不是 ABI 兼容的情况下，调用超类的方法获取输入张量的步幅
            super().codegen_input_stride_var_decl(code, name)
    # 在命名空间内生成 AOTInductorModelKernels 类的代码
    def codegen_model_kernels(self):
        # 输出命名空间声明
        self.prefix.writeline("namespace {")
        # 输出类声明，继承自 AOTInductorModelKernelsBase
        self.prefix.writeline(
            "class AOTInductorModelKernels : public AOTInductorModelKernelsBase {"
        )
        self.prefix.writeline("  public:")
        # 收集所有要声明的内核名称，包括源码到内核的映射值
        declare_kernel = set(self.src_to_kernel.values())
        declare_kernel.update(
            entry[0] for entry in self.user_defined_kernel_cache.values()
        )
        # 如果有常量模块，也收集其内核名称
        if V.graph.const_module:
            declare_kernel.update(
                V.graph.const_module.wrapper_code.src_to_kernel.values()
            )
        # 对内核名称进行排序并声明
        for kernel in sorted(declare_kernel):
            self.prefix.writeline(
                maybe_hipify_code_wrapper(f"    CUfunction {kernel}{{nullptr}};")
            )
        # 输出类定义结尾
        self.prefix.writeline("};")
        # 输出命名空间结束
        self.prefix.writeline("}  // namespace")

    # 根据推断模式生成代码
    def generate(self, is_inference):
        # 如果是 AOT 模式且不是常量图
        if V.graph.aot_mode and not V.graph.is_const_graph:
            # 生成模型内核代码
            self.codegen_model_kernels()
            # 生成模型构造函数代码
            self.codegen_model_constructor()
            # 生成常量运行驱动程序代码
            self.codegen_const_run_driver()
        # 写入包装器声明
        self.write_wrapper_decl()
        # 调用父类的生成方法
        return super().generate(is_inference)

    # 完成前缀生成
    def finalize_prefix(self):
        # 如果配置支持 ABI 兼容性
        cached_dtypes_buffer = IndentedBuffer()
        if config.abi_compatible:
            # 将已使用的缓存数据类型写入缓冲区
            for dtype in self.used_cached_dtypes:
                cached_dtypes_buffer.writeline(f"CACHE_TORCH_DTYPE({dtype});")
            for device in self.used_cached_devices:
                cached_dtypes_buffer.writeline(f"CACHE_TORCH_DEVICE({device});")
            for layout in self.used_cached_layouts:
                cached_dtypes_buffer.writeline(f"CACHE_TORCH_LAYOUT({layout});")
        # 将缓存的数据类型写入前缀
        cached_dtypes_buffer.splice(self.prefix)
        self.prefix = cached_dtypes_buffer

    # 定义内核代码
    def define_kernel(
        self, name: str, kernel: str, metadata: Optional[str] = None, cuda=False
    ):
        # 将内核代码添加到头文件中
        self.header.splice(f"\n{kernel}\n")

    # 生成标量到张量的代码
    def codegen_scalar_to_tensor(self, output: str):
        # 创建标量到张量的转换变量名
        name = f"scalar_to_tensor_{next(self.scalar_to_tensor_id)}"
        # 输出标量到张量的转换代码
        self.wrapper_call.writeline(
            f"RAIIAtenTensorHandle {name} = scalar_to_tensor_handle({output});"
        )
        return name

    # 生成张量元素提取的代码
    def codegen_tensor_item(
        self, dtype: torch.dtype, tensor: str, scalar: str, indented_buffer=None
    ):
        # 断言是否为 ABI 兼容模式
        assert (
            config.abi_compatible
        ), "codegen_tensor_item is only used for the ABI-compatible mode"
        # 获取数据类型的字符串表示
        dtype_str = str(dtype).split(".")[-1]
        # 根据传入的缓冲区或者当前对象输出数据类型的声明
        writer = indented_buffer or self
        writer.writeline(f"{DTYPE_TO_CPP[dtype]} {scalar};")

        # 需要针对 ArrayRefTensors 进行转换
        tensor = f"convert_arrayref_tensor_to_tensor({tensor})"

        # 输出调用元素提取函数的代码
        writer.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_item_{dtype_str}({tensor}, &{scalar}));"
        )

    # 在对象上缓存自身的装饰器
    @cache_on_self
    # 返回一个列表，其中每个元素都是一个字符串表示的引用，根据条件生成不同形式的输出引用
    def get_output_refs(self):
        return [
            # 如果 x 是 ir.ShapeAsConstantBuffer 类型且 config.abi_compatible 不为真，则生成特定格式的 torch 引用
            f"torch::tensor({x.codegen_reference(self.wrapper_call)})" 
            # 否则生成一般的代码引用
            if isinstance(x, ir.ShapeAsConstantBuffer) and not config.abi_compatible 
            else x.codegen_reference(self.wrapper_call)
            for x in V.graph.graph_outputs  # 遍历 V.graph.graph_outputs 中的每个元素 x
        ]

    # 根据条件生成结果之前的后缀代码
    def generate_before_suffix(self, result):
        # 如果不是常量图形
        if not V.graph.is_const_graph:
            # 如果处于 AOT 模式
            if V.graph.aot_mode:
                # 在 result 中写入 AOTInductorModel::run_impl 的结束注释
                result.writeline("} // AOTInductorModel::run_impl")
            else:
                # 在 result 中写入 inductor_entry_impl 的结束注释
                result.writeline("} // inductor_entry_impl")
    def generate_end(self, result):
        if V.graph.aot_mode:
            if V.graph.is_const_graph:
                result.writeline("} // AOTInductorModel::_const_run_impl")
            else:
                result.writeline("} // namespace aot_inductor")
                result.writeline("} // namespace torch")
            return

        # 如果处于 AOT 模式
        result.writeline("'''")  # 写入三个单引号，可能是为了注释多行代码的开始
        result.splice(
            f"""
            inductor_entry = CppWrapperCodeCache.load_pybinding(
                ["std::vector<AtenTensorHandle>"], cpp_wrapper_src, {self.cuda}, {len(V.graph.graph_outputs)})
            """
        )

        # 准备包装器函数体，用于 CPP 包装器的 JIT 入口
        wrapper_body = "input_tensors = [arg if isinstance(arg, torch.Tensor) else torch.tensor(arg) for arg in args]"
        if V.graph.constants:
            # 将常量附加到 CPP 包装器的输入参数中
            # Python 包装器直接获取包装器调用时传递的全局变量的值
            # 作为 exec(code, mod.__dict__, mod.__dict__) 调用时传递的参数
            # 对于 CPP 包装器，我们需要显式地将这个 Python 值传递给 inductor_entry_impl 函数。
            assert all(
                isinstance(v, torch.Tensor) for v in list(V.graph.constants.values())
            ), "Expect all constants to be Tensor"
            constants_str = f"[{', '.join(V.graph.constants.keys())}]"
            wrapper_body += f"""
                    constants_tensor = {constants_str}
                    input_tensors.extend(constants_tensor)
            """
        # 将 at::Tensor 的向量转换为 AtenTensorHandle 的向量。
        # 如果我们传递 at::Tensor，编译速度将会太慢。
        wrapper_body += """
                    input_handles = torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(input_tensors)
        """

        # 将输出张量解包回 Python 标量
        if all(x for x in self.output_is_tensor.values()):
            # 如果输出中没有 ShapeAsConstantBuffer，则直接将输出作为张量返回
            outputs_str = "output_tensors"
        else:
            outputs = [
                f"output_tensors[{i}]"
                if self.output_is_tensor[i]
                else f"output_tensors[{i}].item()"
                for i in range(len(V.graph.graph_outputs))
            ]
            outputs_str = f"[{', '.join(outputs)}]"
        wrapper_body += f"""
                    output_handles = f(input_handles)
                    output_tensors = torch._C._aoti.alloc_tensors_by_stealing_from_void_ptrs(output_handles)
                    return {outputs_str}
        """

        # 包装函数以支持设置 result._boxed_call = True
        result.splice(
            f"""
            def _wrap_func(f):
                def g(args):
                    {wrapper_body}
                return g

            call = _wrap_func(inductor_entry)
            """
        )
    def get_c_shim_func_name(self, kernel):
        # 如果不是 ABI 兼容模式或者 kernel 以 "aoti_torch_" 开头，则直接返回 kernel
        if not config.abi_compatible or kernel.startswith("aoti_torch_"):
            return kernel
        
        # 确保 kernel 中包含 "::"，用于分割 kernel 名称
        assert "::" in kernel, "Cpp kernel name: " + kernel + " does not contain '::'"
        kernel_tokens = kernel.split("::")
        kernel_suffix = kernel_tokens[-1]
        # 如果 kernel 后缀为 "call"，则将其替换为倒数第二个 token
        if kernel_suffix == "call":
            kernel_suffix = kernel_tokens[-2]
        
        # 根据配置选择 C shim 的版本
        if config.c_shim_version == "1":
            # 对于 sdpa，需要选择 v2 版本，因为 v1 版本不考虑可选参数
            if kernel_suffix == "_scaled_dot_product_flash_attention":
                shim_fn = "aoti_torch__scaled_dot_product_flash_attention_v2"
            # 对于以 "wrapped_fbgemm" 开头的 kernel，需要特殊处理
            elif kernel_suffix.startswith("wrapped_fbgemm"):
                # 确保在 CPU 上使用 wrapped_fbgemm
                assert self.device == "cpu", "Using wrapped_fbgemm out of CPU!"
                shim_fn = f"aoti_torch_cpu_{kernel_suffix}"
            else:
                shim_fn = f"aoti_torch_{kernel_suffix}"
        else:
            # 使用特定设备和 kernel 后缀生成 C shim 函数名称
            shim_fn = f"aoti_torch_{self.device}_{kernel_suffix}"
        
        return shim_fn

    def generate_c_shim_extern_kernel_call(self, kernel, args):
        # 在 abi_compatible 模式下，通过 C shim 层调用回退的 aten 操作
        # 将 self.allow_stack_allocation 设置为 False，因为 ArrayRefTensor 和 at::Tensor 之间的交换仍然不稳定。
        self.allow_stack_allocation = False

        wrapped_args = []
        for x in args:
            pieces = x.split(", ")
            for piece in pieces:
                # 只有真正需要的时候才会转换 ArrayRefTensor 到 Tensor
                # 对于整数直接跳过，避免错误地将其视为指针
                if isinstance(piece, str) and piece.startswith(
                    ("buf", "arg", "wrap_with_raii_handle_if_needed")
                ):
                    piece = f"convert_arrayref_tensor_to_tensor({piece})"
                wrapped_args.append(piece)

        # 获取 C shim 函数名称
        shim_fn = self.get_c_shim_func_name(kernel)
        # 生成调用 C shim 外部 kernel 的代码行
        self.writeline(
            f"AOTI_TORCH_ERROR_CODE_CHECK({shim_fn}({', '.join(wrapped_args)}));"
        )

    def generate_c_shim_extern_kernel_alloc(self, extern_kernel, args):
        # 注册输出缓冲区名称
        name = extern_kernel.name
        output_handle_name = f"{name}_handle"
        # 声明 AtenTensorHandle 并赋值为输出句柄名称的地址
        self.writeline(f"AtenTensorHandle {output_handle_name};")
        output_arg = f"&{output_handle_name}"
        # 调用生成 C shim 外部 kernel 调用的方法，传入参数列表和输出参数
        self.generate_c_shim_extern_kernel_call(
            extern_kernel.get_kernel_name(), args + [output_arg]
        )
        # 声明 RAIIAtenTensorHandle 并使用输出句柄名称进行初始化
        self.writeline(f"RAIIAtenTensorHandle {name}({output_handle_name});")
    # 如果配置为 ABI 兼容，则调用生成 C 语言的外部内核分配方法
    def generate_extern_kernel_alloc(self, extern_kernel, args):
        if config.abi_compatible:
            # 调用生成 C 语言的外部内核分配方法
            self.generate_c_shim_extern_kernel_alloc(extern_kernel, args)
        else:
            # 否则调用父类的外部内核分配方法
            super().generate_extern_kernel_alloc(extern_kernel, args)

    # 生成 C 语言的回退内核方法
    def generate_c_shim_fallback_kernel(self, fallback_kernel, args):
        output_args = []
        output_raii_handles = []
        output_name_base = fallback_kernel.get_name()
        for idx, output in enumerate(fallback_kernel.outputs):
            if isinstance(output, ir.MultiOutput):
                # TODO: 处理整数输出（例如，如在注意力中）
                name = f"{output.get_name()}"
                output_handle_name = f"{name}_handle"
                if output.indices:
                    assert (
                        output.indices[0][1] == idx
                    ), f"expected {output.indices[0][1]=} == {idx=} for {output_name_base=}"
                self.writeline(f"AtenTensorHandle {output_handle_name};")
                output_args.append(f"&{output_handle_name}")
                output_raii_handles.append(
                    f"RAIIAtenTensorHandle {name}({output_handle_name});"
                )
            elif isinstance(output, int):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"int64_t {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif isinstance(output, sympy.Symbol):
                output_name = f"{output_name_base}_{idx}"
                self.writeline(f"auto {output_name} = {output};")
                output_args.append(f"&{output_name}")
            elif output is None:
                output_args.append("nullptr")
            else:
                # 抛出未实现的错误类型
                raise NotImplementedError(f"unsupported type of {output=}")
        args = args + output_args
        # 调用生成 C 语言的外部内核调用方法
        self.generate_c_shim_extern_kernel_call(fallback_kernel.cpp_kernel_name, args)
        for raii_handle in output_raii_handles:
            self.writeline(raii_handle)

    # 生成回退内核方法
    def generate_fallback_kernel(self, fallback_kernel, args):
        if config.abi_compatible:
            # 调用生成 C 语言的回退内核方法
            self.generate_c_shim_fallback_kernel(fallback_kernel, args)
        else:
            # 否则调用父类的回退内核方法
            super().generate_fallback_kernel(fallback_kernel, args)

    # 生成外部内核输出方法
    def generate_extern_kernel_out(
        self, kernel: str, out: str, out_view: Optional[str], args: List[str]
    ):
        if out_view:
            out_name = f"{out}_as_strided"
            self.writeline(f"auto {out_name} = {out_view};")
            args.insert(0, out_name)
        else:
            args.insert(0, out)

        if config.abi_compatible:
            # 调用生成 C 语言的外部内核调用方法
            self.generate_c_shim_extern_kernel_call(kernel, args)
        else:
            # 否则包装内核调用并写入
            self.writeline(self.wrap_kernel_call(kernel, args))

    # 生成散列回退方法
    def generate_scatter_fallback(
        self,
        output,
        inputs,
        cpp_kernel_name,
        python_kernel_name,
        src_is_tensor,
        reduce,
        kwargs,
        # 如果存在回退操作，则不允许堆栈分配
        self.allow_stack_allocation = False

        # TODO: 需要更新以使用 C 语言桥接层 v2
        if config.abi_compatible:
            # 使用 ABI 桥接函数替代 ATen 函数
            if config.c_shim_version == "1":
                # 根据函数名前缀选择对应的 C++ 内核函数
                cpp_kernel_name = (
                    "aoti_torch_scatter_reduce_out"
                    if python_kernel_name.startswith("aten.scatter_reduce")
                    else "aoti_torch_scatter_out"
                )
            else:
                # 获取适配的 C 语言桥接函数名
                cpp_kernel_name = self.get_c_shim_func_name(cpp_kernel_name)
                # C 语言桥接函数仅包含 out-variant，不支持 inplace-variant
                cpp_kernel_name = cpp_kernel_name.replace("__", "_") + "_out"
            
            # 将输入参数转换为对应的包装形式
            inputs_wrapped = [
                f"convert_arrayref_tensor_to_tensor({x})"
                if isinstance(x, str)
                else str(x)
                for x in inputs
            ]
            # 构造 C++ 内核函数调用的代码行
            line = f"{cpp_kernel_name}(convert_arrayref_tensor_to_tensor({output}), {','.join(inputs_wrapped)}"
        else:
            # 构造普通情况下的 C++ 内核函数调用的代码行
            line = f"{cpp_kernel_name}({','.join(map(str, inputs))}"
        
        # 如果 Python 内核函数名以 "aten.scatter_reduce" 开头，则添加额外的参数
        if python_kernel_name.startswith("aten.scatter_reduce"):
            line += f", {','.join(kwargs)}"
        else:
            # 如果源张量是张量，则根据情况添加 reduce 参数
            if src_is_tensor:
                if reduce:
                    line += f", {V.graph.wrapper_code.val_to_arg_str(reduce)}"
            else:
                # 对于标量源张量的 aten.scatter_，预期 reduce 参数为 None
                assert (
                    reduce is None
                ), "Expect reduce to be None for aten.scatter_ with scalar src"
        
        # 结束 C++ 内核函数调用的代码行
        line += ");"
        # 将构造好的代码行写入到代码生成器中
        self.writeline(line)
    # 设置标志以禁止在出现回退操作时分配堆栈空间
    self.allow_stack_allocation = False

    # TODO: 需要更新以使用 C 语言接口 v2
    if config.abi_compatible:
        # 在 abi_compatible 模式下，创建一个包含 indices 的 AtenTensorHandle 的 std::vector 对象，并获取其指针
        indices_str = (
            "std::vector<AtenTensorHandle>{"
            + (
                ", ".join(
                    [f"convert_arrayref_tensor_to_tensor({ind})" for ind in indices]
                )
            )
            + "}.data()"
        )
        # 准备调用的参数列表，包括将 x 和 values 转换为张量的步骤
        args = [
            f"convert_arrayref_tensor_to_tensor({x})",
            indices_str,
            str(len(indices)),
            f"convert_arrayref_tensor_to_tensor({values})",
            accumulate,
        ]
        # 将 x 插入到参数列表的开头，以便将其用作输出张量，这个回退操作会改变 x
        args.insert(
            0, f"convert_arrayref_tensor_to_tensor({x})"
        )
    else:
        # 生成 indices 的字符串表示，用于非 abi_compatible 模式
        indices_str = (
            f"{self.open_bracket}{', '.join(indices)}{self.closed_bracket}"
        )
        # 准备调用的参数列表，直接使用 x、indices_str、values 和 accumulate
        args = [x, indices_str, values, accumulate]
        # 将 x 插入到参数列表的开头，以便将其用作输出张量，这个回退操作会改变 x

    # 将包装后的 kernel 调用写入输出
    self.writeline(self.wrap_kernel_call(kernel, args))


    # 如果处于 V.graph.aot_mode 模式下，不添加基准测试工具
    if V.graph.aot_mode:
        return

    # 调用父类的方法添加基准测试工具到输出
    super().add_benchmark_harness(output)


    # 为给定表达式 x 生成其代码表示，并返回字符串形式
    return self.expr_printer(V.graph.sizevars.simplify(x))


    # 根据给定的 basename、name 和 index 生成 tuple 访问的代码表示
    if config.abi_compatible:
        # 在 abi_compatible 模式下，通过参数返回输出
        return name
    else:
        # 在非 abi_compatible 模式下，使用 std::get 获取 tuple 中指定索引的元素
        return f"std::get<{index}>({basename})"


    # 根据给定的 shape 元组生成其代码表示的字符串形式
    parts = list(map(self.codegen_sizevar, shape))
    if len(parts) == 0:
        return "{}"
    if len(parts) == 1:
        return f"{{{parts[0]}, }}"
    return f"{{{', '.join(parts)}}}"
    # 生成动态标量的代码。根据节点的输入生成数据引用。
    def codegen_dynamic_scalar(self, node):
        # 使用生成器表达式从节点的输入中获取数据引用
        (data,) = (t.codegen_reference() for t in node.inputs)
        
        # 如果配置允许 ABI 兼容性
        if config.abi_compatible:
            # 调用 codegen_tensor_item 方法处理张量项
            self.codegen_tensor_item(
                node.inputs[0].get_dtype(), data, f"{node.sym}_raw"
            )
        else:
            # 根据节点输入的数据类型，选择对应的 ATEN 函数进行类型转换
            convert_type = DTYPE_TO_ATEN[node.inputs[0].get_dtype()].replace(
                "at::k", "to"
            )
            self.writeline(f"auto {node.sym}_raw = {data}.item().{convert_type}();")
        
        # 如果节点的 keypath 长度为 0，直接将 raw 数据赋给 sym
        if len(node.keypath) == 0:
            self.writeline(f"auto {node.sym} = {node.sym}_raw;")
        # 如果 keypath 长度为 1 并且 keypath[0] 是 ConvertIntKey 类型
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], ConvertIntKey):
            # 将 raw 数据转换为 int64_t 类型，根据条件设置为 1 或 0
            self.writeline(f"int64_t {node.sym} = {node.sym}_raw ? 1 : 0;")
        # 如果 keypath 长度为 1 并且 keypath[0] 是 DivideByKey 类型
        elif len(node.keypath) == 1 and isinstance(node.keypath[0], DivideByKey):
            # TODO: 在此处验证可除性
            # 将 raw 数据除以 keypath[0] 中的除数，赋给 sym
            self.writeline(
                f"int64_t {node.sym} = {node.sym}_raw / {node.keypath[0].divisor};"
            )
        else:
            # 如果 keypath 不符合已知的类型，抛出断言错误
            raise AssertionError(f"unrecognized keypath {node.keypath}")
        
        # 将符号添加到 unbacked_symbol_decls 集合，以避免再次生成声明
        self.unbacked_symbol_decls.add(str(node.sym))

    # 检查是否可以在栈上分配缓冲区
    def can_stack_allocate_buffer(self, buffer):
        return (
            self.allow_stack_allocation
            and buffer.get_device().type == "cpu"
            and self.can_prove_buffer_has_static_shape(buffer)
            and ir.is_contiguous_strides_for_shape(
                buffer.get_stride(), buffer.get_size()
            )
        )

    # 生成释放缓冲区的代码
    def make_buffer_free(self, buffer):
        return (
            ""
            if isinstance(buffer.get_layout(), ir.MultiOutputLayout)
            or (V.graph.aot_mode and buffer.get_name() in self.stack_allocated_buffers)
            or (
                config.use_minimal_arrayref_interface
                and V.graph.aot_mode
                and buffer.get_name() in V.graph.graph_inputs
            )
            else f"{buffer.get_name()}.reset();"
        )

    # 根据名称列表生成释放操作的代码
    def make_free_by_names(self, names_to_del: List[str]):
        return " ".join(f"{name}.reset();" for name in names_to_del)

    # 生成确切的缓冲区重用代码
    def codegen_exact_buffer_reuse(self, old_name: str, new_name: str, del_line: str):
        # 如果配置允许 ABI 兼容性
        if config.abi_compatible:
            # 使用 std::move 将 old_name 的内容移动到 new_name 中，表示重用
            return f"auto {new_name} = std::move({old_name});  // reuse"
        else:
            # 否则调用父类的方法处理确切的缓冲区重用
            return super().codegen_exact_buffer_reuse(old_name, new_name, del_line)

    # 生成调用包装器的性能分析标记代码
    def generate_profiler_mark_wrapper_call(self, stack):
        # 在包装器调用中记录函数执行，使用 c10::ArrayRef<c10::IValue> 作为参数
        self.wrapper_call.writeline(
            'RECORD_FUNCTION("inductor_wrapper_call", c10::ArrayRef<c10::IValue>());'
        )

    # 写入 Triton 头文件的占位方法
    def write_triton_header_once(self):
        pass

    # 生成开始图形的占位方法
    def generate_start_graph(self):
        pass

    # 生成结束图形的占位方法
    def generate_end_graph(self):
        pass
    def generate_inf_and_nan_checker(self, nodes):
        for buf in nodes.get_names():
            # 遍历节点名称，将节点名称直接添加到检查无穷大和 NaN 的函数调用中
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_check_inf_and_nan({buf}));"
            )

    def codegen_device(self, device):
        if config.abi_compatible:
            # 如果与 ABI 兼容，将设备类型添加到已使用的缓存设备集合中，并返回缓存的 Torch 设备类型和索引
            self.used_cached_devices.add(device.type)
            return f"cached_torch_device_type_{device.type}, {device.index if device.index else 0}"
        else:
            # 如果不与 ABI 兼容，根据设备类型和索引生成 c10::Device 对象
            return (
                f"c10::Device({DEVICE_TO_ATEN[device.type]}, {device.index if device.index is not None else 0})"
                if device.index is not None
                else f"{DEVICE_TO_ATEN[device.type]}"
            )

    def codegen_dtype(self, dtype):
        if config.abi_compatible:
            # 如果与 ABI 兼容，将 dtype 转换为字符串形式，并将其添加到已使用的缓存 dtype 集合中，返回缓存的 Torch dtype 字符串
            dtype_str = str(dtype).split(".")[-1]
            self.used_cached_dtypes.add(dtype_str)
            return f"cached_torch_dtype_{dtype_str}"
        else:
            # 如果不与 ABI 兼容，根据 dtype 返回对应的 ATen dtype
            return DTYPE_TO_ATEN[dtype]

    def codegen_layout(self, layout):
        if config.abi_compatible:
            # 如果与 ABI 兼容，将 layout 转换为字符串形式，并将其添加到已使用的缓存 layout 集合中，返回缓存的 Torch layout 字符串
            layout_str = str(layout).split(".")[-1]
            self.used_cached_layouts.add(layout_str)
            return f"cached_torch_layout_{layout_str}"
        else:
            # 如果不与 ABI 兼容，根据 layout 返回对应的 ATen layout
            return LAYOUT_TO_ATEN[layout]

    @functools.lru_cache(None)  # noqa: B019
    def codegen_int_array_var(
        self,
        int_array: str,
        writer=None,
        known_statically=False,
        graph=None,  # for per-graph caching
    ):
        # 用于声明大小/步幅的整数数组
        # 由于内存规划分两次进行（参见 self.generate 的实现），
        # 在两次遍历中，writeline 的行为不同。
        # 因此，生成的 int 数组声明可能出现在生成代码的较后位置，
        # 因此第二次遍历的 codegen 不应重用第一次遍历生成的 int 数组声明
        if writer is None:
            # 第一次遍历的 codegen 使用 self 作为写入器
            writer = self

        var = f"int_array_{next(self.int_array_id)}"
        ctype = "int64_t"
        if var not in self.declared_int_array_vars:
            self.declared_int_array_vars.add(var)
            if known_statically:
                # 如果已知静态，生成静态的 constexpr int 数组声明
                writer.writeline(f"static constexpr {ctype} {var}[] = {int_array};")
            else:
                # 否则生成 const int 数组声明
                writer.writeline(f"const {ctype} {var}[] = {int_array};")
        return var

    def make_buffer_allocation(self, buffer):
        # 创建缓冲区分配
        return self.make_allocation(
            buffer.get_name(),
            buffer.get_device(),
            buffer.get_dtype(),
            buffer.get_size(),
            buffer.get_stride(),
            buffer if self.can_stack_allocate_buffer(buffer) else None,
        )

    def make_allocation(
        self, name, device, dtype, shape, stride, buffer_if_can_stack_allocate=None
    ):
        # 创建分配
        ):
        # 保存原始的步长值
        orig_stride = stride
        # 将设备字符串转换为特定设备的代码表示形式
        device_str = self.codegen_device(device)
        # 将数据类型转换为特定数据类型的代码表示形式
        dtype_code = self.codegen_dtype(dtype)
        # 将形状转换为元组的代码表示形式
        size = self.codegen_shape_tuple(shape)
        # 将原始步长转换为元组的代码表示形式
        stride = self.codegen_shape_tuple(orig_stride)
        # 如果配置兼容 ABI
        if config.abi_compatible:
            # 生成表示尺寸的整数数组变量
            size_array_var = self.codegen_int_array_var(
                size,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(shape),
                graph=self.get_codegened_graph(),
            )
            # 生成表示步长的整数数组变量
            stride_array_var = self.codegen_int_array_var(
                stride,
                self.wrapper_call,
                known_statically=self.is_statically_known_list_of_ints(orig_stride),
                graph=self.get_codegened_graph(),
            )
            # 解析设备类型和设备 ID
            device_type, device_id = device_str.split(",")
            # 如果处于 AOT 模式，设备索引使用特定的表达方式
            device_idx = "this->device_idx_" if V.graph.aot_mode else device_id
            # 如果可以在栈上分配缓冲区
            if buffer_if_can_stack_allocate is not None:
                # 将可栈分配的缓冲区保存到字典中
                self.stack_allocated_buffers[name] = buffer_if_can_stack_allocate
                # 确定 C++ 类型
                cpp_type = DTYPE_TO_CPP[dtype]
                # 获取缓冲区中元素的数量
                numel = buffer_if_can_stack_allocate.get_numel()
                # 输出 C++ 代码，声明具有指定元素数量的数组
                self.wrapper_call.writeline(f"{cpp_type} {name}_storage[{numel}];")
                # 构建参数列表
                args = [
                    f"{name}_storage",
                    size_array_var,
                    stride_array_var,
                    device_type,
                    device_idx,
                ]
                # 返回 C++ 代码，创建 ArrayRefTensor 对象
                return f"ArrayRefTensor<{cpp_type}> {name}({', '.join(args)});"

            # 构建参数列表
            args = [
                str(len(shape)),
                size_array_var,
                stride_array_var,
                dtype_code,
                device_type,
                device_idx,
                f"&{name}_handle",
            ]

            # 输出 C++ 代码，声明 AtenTensorHandle 对象
            self.wrapper_call.writeline(f"AtenTensorHandle {name}_handle;")
            # 输出 C++ 代码，调用 AOT 接口生成张量
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided({', '.join(args)}));"
            )

            # 返回 C++ 代码，创建 RAIIAtenTensorHandle 对象
            return f"RAIIAtenTensorHandle {name}({name}_handle);"

        # 如果处于 AOT 模式并且设备字符串以 "c10::Device(" 开头
        if V.graph.aot_mode and device_str.startswith("c10::Device("):
            # 构建张量的设备字符串
            tensor_device = f"{device_str.split(',')[0]}, this->device_idx_)"
        else:
            # 否则使用原始的设备字符串
            tensor_device = device_str

        # 根据设备类型生成不同的张量创建代码
        if device.type == "cpu":
            return f"at::Tensor {name} = at::detail::empty_strided_cpu({size}, {stride}, {dtype_code});"
        if device.type == "cuda":
            return (
                f"at::Tensor {name} = at::detail::empty_strided_cuda("
                f"{size}, {stride}, {dtype_code}, c10::DeviceType::CUDA);"
            )
        # 返回默认的张量创建代码
        return (
            f"{self.declare}{name} = {self.namespace}empty_strided("
            f"{size}, {stride}, at::TensorOptions({tensor_device}).dtype({dtype_code})){self.ending}"
        )
    # 从对象池中分配代码生成的张量句柄，返回句柄名称字符串
    def codegen_alloc_from_pool(self, name, offset, dtype, shape, stride) -> str:
        # 如果配置为 ABI 兼容模式
        if config.abi_compatible:
            # 计算形状和步长的元组大小
            size = self.codegen_shape_tuple(shape)
            stride = self.codegen_shape_tuple(stride)
            # 创建临时张量句柄名称
            tmp_name = f"tmp_tensor_handle_{next(self.tmp_tensor_id)}"
            # 准备函数调用的参数列表
            args = [
                name,
                self.expr_printer(offset),  # bytes not numel
                self.codegen_dtype(dtype),
                str(len(shape)),
                self.codegen_int_array_var(
                    size, self.wrapper_call, graph=self.get_codegened_graph()
                ),
                self.codegen_int_array_var(
                    stride, self.wrapper_call, graph=self.get_codegened_graph()
                ),
                f"&{tmp_name}",
            ]
            # 在代码生成器中写入声明张量句柄的语句
            self.wrapper_call.writeline(f"AtenTensorHandle {tmp_name};")
            # 在代码生成器中写入调用错误检查的语句
            self.wrapper_call.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch__alloc_from_pool({', '.join(args)}));"
            )
            # 返回使用 RAII 封装的张量句柄对象字符串
            return f"RAIIAtenTensorHandle({tmp_name})"

        # 如果不是 ABI 兼容模式，则生成普通的分配函数调用字符串
        return "alloc_from_pool({})".format(
            ", ".join(
                [
                    name,
                    self.expr_printer(offset),  # bytes not numel
                    self.codegen_dtype(dtype),
                    self.codegen_shape_tuple(shape),
                    self.codegen_shape_tuple(stride),
                ]
            )
        )

    # 生成重解释视图的代码
    def codegen_reinterpret_view(
        self, data, size_list, stride_list, offset, writer
    ):
        # 实现待补充
        pass

    # 生成设备间复制的代码
    def codegen_device_copy(self, src, dst):
        # 如果配置为 ABI 兼容模式
        if config.abi_compatible:
            # aoti_torch_tensor_copy_ 接受 AtenTensorHandle 作为输入，
            # 而堆栈分配导致 ArrayRefTensor，因此在此禁用堆栈分配
            self.allow_stack_allocation = False
            # 在代码生成器中写入复制函数调用的错误检查语句
            self.writeline(
                f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_tensor_copy_(expensive_copy_to_tensor_if_needed({src}), {dst}));"
            )
        else:
            # 在代码生成器中写入普通的复制调用语句
            self.writeline(f"{dst}.copy_({src});")

    # 生成多输出的代码
    def codegen_multi_output(self, name, value):
        # 如果不是 ABI 兼容模式，则调用父类的多输出代码生成方法
        if not config.abi_compatible:
            super().codegen_multi_output(name, value)
    def codegen_subgraph_prefix(self, subgraph, outer_inputs, outer_outputs):
        # 遍历子图的输入和外部输入，将它们进行关联
        for inner_input, outer_input in zip(subgraph.graph.graph_inputs, outer_inputs):
            if config.abi_compatible:
                # 如果处于ABI兼容模式，则将外部输入（outer_input）的底层at::Tensor复制到另一个at::Tensor中，
                # 作为嵌套范围内部输入（inner_input）的子图输入使用。这里不能使用std::move，
                # 因为codegened的外部输入可能是表达式或rvalue（例如reinterpret_view(x)），
                # 所以不能将其std::move回原始位置（x）。
                self.writeline(f"AtenTensorHandle {inner_input}_handle;")
                self.writeline(
                    f"AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_assign_tensors_out({outer_input}, &{inner_input}_handle));"
                )
                self.writeline(
                    f"RAIIAtenTensorHandle {inner_input}({inner_input}_handle);"
                )
            else:
                # 如果不处于ABI兼容模式，则直接赋值外部输入给内部输入
                self.writeline(
                    f"{self.declare}{inner_input} = {outer_input}{self.ending}"
                )

    def codegen_subgraph_suffix(self, subgraph, outer_inputs, outer_outputs):
        # 遍历子图的输出和外部输出，将它们进行关联
        for inner_output, outer_output in zip(
            subgraph.graph.graph_outputs, outer_outputs
        ):
            src = inner_output.codegen_reference()
            if config.abi_compatible:
                # 如果处于ABI兼容模式，则需要将子图输出（inner_output）std::move到外部输出（outer_output），
                # 因为RAIIAtenTensorHandle的复制构造函数已删除。
                src = f"std::move({src})"
                # 在外部输出之前，可能存在值（例如在while_loop的codegen中）
                self.writeline(f"{outer_output}.reset();")
            # 将源（src）赋值给外部输出（outer_output）
            self.writeline(f"{outer_output} = {src}{self.ending}")
    # 生成条件语句的代码，处理条件对象的代码生成
    def codegen_conditional(self, conditional):
        # 获取条件对象的名称
        name = conditional.get_name()
        # 获取条件操作数的代码生成引用列表
        outer_inputs = [f"{buf.codegen_reference()}" for buf in conditional.operands]
        
        # 如果配置为 ABI 兼容模式
        if config.abi_compatible:
            # 外部输出变量列表初始化为空
            outer_outputs = []
            # 遍历条件语句的输出对象
            for out in conditional.outputs:
                # 在 ABI 兼容模式下，ir.MultiOutput 不会被代码生成，
                # 因此直接预声明输出变量
                self.writeline(f"RAIIAtenTensorHandle {out.get_name()};")
                outer_outputs.append(out.get_name())

            # 如果条件判断不是 ir.ShapeAsConstantBuffer 类型
            if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
                # 在 ABI 兼容模式下，需要使用 ABI 适配函数
                # 从底层标量 bool Tensor 中提取 C++ bool
                predicate = f"{conditional.predicate.get_name()}_scalar"
                self.codegen_tensor_item(
                    torch.bool,
                    conditional.predicate.codegen_reference(),
                    predicate,
                )
            else:
                # 条件判断不是 Tensor：SymBool 或 Python bool
                predicate = conditional.predicate.codegen_reference()
        else:
            # 非 ABI 兼容模式下，可以将条件输出代码生成为 at::Tensor 实例数组，
            # 因为 ir.MultiOutput 会被代码生成
            outer_outputs = [f"{name}[{i}]" for i in range(len(conditional.outputs))]
            self.writeline(f"at::Tensor {name}[{len(conditional.outputs)}];")
            predicate = f"{conditional.predicate.codegen_reference()}"
            # 如果条件判断不是 ir.ShapeAsConstantBuffer 类型
            if not isinstance(conditional.predicate, ir.ShapeAsConstantBuffer):
                # 将 Tensor 条件移动到主机
                predicate = f"{predicate}.item<bool>()"

        # 输出条件语句的条件判断
        self.writeline(f"if ({predicate}) {{")
        # 进入条件语句的 true 子图代码生成
        self.writeline(EnterSubgraphLine(self, conditional.true_subgraph.graph))
        self.codegen_subgraph(conditional.true_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("} else {")
        # 进入条件语句的 false 子图代码生成
        self.writeline(EnterSubgraphLine(self, conditional.false_subgraph.graph))
        self.codegen_subgraph(conditional.false_subgraph, outer_inputs, outer_outputs)
        self.writeline(ExitSubgraphLine(self))
        self.writeline("}")
    
    # 如果需要，生成外部内核参数声明的代码
    def generate_extern_kernel_args_decl_if_needed(
        self, op_overload, raw_args, output_args
    ):
        pass
    
    # 如果需要，生成外部内核分配和查找架构的代码
    def generate_extern_kernel_alloc_and_find_schema_if_needed(
        self,
        buf_name: str,
        python_kernel_name: str,
        cpp_kernel_name: str,
        codegen_args: List[str],
        cpp_op_schema: str,
        cpp_kernel_key: str,
        cpp_kernel_overload_name: str = "",
        op_overload: Optional[torch._ops.OpOverload] = None,
        raw_args=None,
        outputs=None,
    ):
        pass
        # 如果存在回退操作，则禁用堆栈分配
        self.allow_stack_allocation = False

        # 定义一个内部函数，用于从输出中提取名称
        def extract_output_name(out):
            # 断言输出不为 None，即不支持可选输出为 None 的情况
            assert out is not None, "None, i.e. optional output is not supported"
            if isinstance(out, (ir.MultiOutput, ir._CollectiveKernel)):
                return out.get_name()  # 如果输出是 MultiOutput 或 CollectiveKernel 类型，返回其名称
            elif isinstance(out, (list, tuple)):
                # 如果输出是列表或元组，则递归提取每个元素的名称
                return type(out)(extract_output_name(o) for o in out)
            else:
                # 如果遇到意外的输出类型，则抛出断言错误
                raise AssertionError(f"Unexpected output: {type(out)}")

        # 如果配置兼容 ABI
        output_args = None
        if config.abi_compatible:
            output_args = extract_output_name(outputs)  # 提取输出的名称作为输出参数
            if isinstance(output_args, str):
                output_args = [output_args]  # 如果输出参数是字符串，则转换为单元素列表

        # 如果配置是 fbcode
        if config.is_fbcode():
            assert op_overload is not None  # 断言操作重载不为 None
            assert raw_args is not None  # 断言原始参数不为 None
            assert outputs is not None  # 断言输出不为 None

            # 调用专门为 fbcode 生成外部内核分配并查找模式的方法
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_fbcode(
                cpp_kernel_key,
                op_overload,
                raw_args,
                output_args,
            )
        else:
            # 否则，调用专门为 oss 生成外部内核分配并查找模式的方法
            return self.generate_extern_kernel_alloc_and_find_schema_if_needed_oss(
                buf_name,
                python_kernel_name,
                cpp_kernel_name,
                codegen_args,
                cpp_op_schema,
                cpp_kernel_key,
                cpp_kernel_overload_name,
                op_overload,
                raw_args,
                output_args,
            )

    # 生成带作用域的 GIL 获取方法
    def generate_scoped_gil_acquire(self, declarations_before_scope, lines_in_scope):
        scoped_lines = IndentedBuffer()  # 创建一个带缩进的缓冲区对象

        # 将作用域前的声明写入缓冲区
        for declaration in declarations_before_scope:
            scoped_lines.writeline(declaration)

        # 添加作用域的起始大括号
        scoped_lines.writeline("{")
        with scoped_lines.indent():  # 使用缓冲区的缩进功能
            scoped_lines.writeline("py::gil_scoped_acquire acquire;")  # 获取 GIL 的 C++ 绑定
            scoped_lines.writelines(lines_in_scope.split("\n"))  # 将作用域内的代码行写入缓冲区
        scoped_lines.writelines("}")  # 添加作用域的结束大括号
        return scoped_lines._lines  # 返回缓冲区中的所有行

    # 加载自定义操作包装器
    def load_custom_op_wrapper(self):
        # TODO: 需要支持控制流，暂时留空以后完善
        if self.custom_op_wrapper_loaded:
            return  # 如果自定义操作包装器已加载，则直接返回

        lines = """
// 使用 PyImport_ImportModule 函数导入名为 "torch._inductor.codecache" 的 Python 模块，并封装为 RAIIPyObject 对象
RAIIPyObject codecache_module(PyImport_ImportModule("torch._inductor.codecache"));
// 检查导入的模块对象是否为 NULL，若为 NULL 则抛出运行时错误
if (codecache_module.get() == NULL) {
    throw std::runtime_error("Failed to load torch._inductor.codecache");
}
// 从 codecache_module 中获取名为 "custom_op_wrapper" 的属性对象，并封装为 PyObject 对象
custom_op_wrapper = PyObject_GetAttrString(codecache_module, "custom_op_wrapper");
// 检查获取的 custom_op_wrapper 对象是否为 NULL，若为 NULL 则抛出运行时错误
if (custom_op_wrapper.get() == NULL) {
    throw std::runtime_error("Failed to load torch._inductor.codecache.custom_op_wrapper");
}
        # 定义生成外部内核分配和查找模式的方法，如果需要的话使用 OSS 版本
        # 参数说明：
        #   buf_name: 缓冲区名称，用于存储操作结果
        #   python_kernel_name: Python 内核名称
        #   cpp_kernel_name: C++ 内核名称
        #   codegen_args: 代码生成参数列表
        #   cpp_op_schema: C++ 内核模式
        #   cpp_kernel_key: C++ 内核键值
        #   cpp_kernel_overload_name: C++ 内核重载名称（可选）
        #   op_overload: Torch 操作重载对象（可选）
        #   raw_args: 原始参数（可选）
        #   output_args: 输出参数列表（可选）

        # 如果处于 AOT 模式或者不兼容 ABI，则使用 OSS 版本的 ProxyExecutor 更新
        if V.graph.aot_mode or not config.abi_compatible:
            # 如果该 C++ 内核键值不在 extern_call_ops 集合中，进行以下操作
            if cpp_kernel_key not in self.extern_call_ops:
                # 写入静态声明，调用 c10::Dispatcher 的 singleton 方法
                self.writeline(
                    f"static auto op_{cpp_kernel_key} = c10::Dispatcher::singleton()"
                )
                # 查找或抛出指定的 C++ 内核名称和重载名称的模式
                self.writeline(
                    f'\t.findSchemaOrThrow("{cpp_kernel_name}", "{cpp_kernel_overload_name}")'
                )
                # 将找到的模式类型化为指定的 C++ 内核模式
                self.writeline(f"\t.typed<{cpp_op_schema}>();")
                # 将当前的 C++ 内核键值添加到 extern_call_ops 集合中
                self.extern_call_ops.add(cpp_kernel_key)

            # 调用已声明的 C++ 内核对象并存储结果到 buf_name 中
            self.writeline(
                f"auto {buf_name} = op_{cpp_kernel_key}.call({', '.join(codegen_args)});"
            )
        else:
            # 在 JIT 模式下，由于 ABI 兼容要求，不能直接调用 c10::Dispatcher 查找自定义操作并调用它
            # 因此回到 Python 环境中执行此自定义操作
            self.load_custom_op_wrapper()

            # 断言确保输出参数列表不为空
            assert output_args is not None, "output_args should not be None"
            # 计算原始参数的数量
            num_args = len(raw_args)
            # 创建一个 Python 参数变量的名称
            py_args_var = f"py_args_{next(self.arg_var_id)}"
            # 第一个参数始终是 Python 操作的名称
            lines = f"""
// 创建一个 PyTuple 对象，用于存储 Python 函数调用的参数
RAIIPyObject {py_args_var}(PyTuple_New({num_args+1}));
// 检查 PyTuple 对象是否创建成功，如果失败则抛出异常
if ({py_args_var}.get() == NULL) {{
    throw std::runtime_error("PyTuple_New {py_args_var} failed");
}}
// 将第一个参数设置为指定的 Python 内核名称
PyTuple_SetItem({py_args_var}, 0, PyUnicode_FromString("{python_kernel_name}"));
"""

// 确保 op_overload 不为 None
assert op_overload is not None, "op_overload should not be None"
// 遍历原始参数和对应的模式化参数架构
for idx, (raw_arg, schema_arg) in enumerate(
    zip(raw_args, op_overload._schema.arguments)
):
    // 生成 Python 参数并添加到 PyTuple 对象中
    lines += self.generate_py_arg(
        py_args_var, idx + 1, raw_arg, schema_arg.real_type
    )

// 在 Python 中调用自定义操作
RAIIPyObject py_{buf_name}(PyObject_CallObject(custom_op_wrapper, {py_args_var}));
// 检查 Python 函数调用是否成功，如果失败则抛出异常
if (py_{buf_name}.get() == NULL) {{
    throw std::runtime_error("PyObject_CallObject {python_kernel_name} failed");
}}

// 如果输出参数只有一个张量
if (len(output_args) == 1) {
    // 将 Python 返回的对象转换为 AtenTensorHandle 类型，并赋值给对应的变量
    lines += f"""
{output_args[0]} = reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(py_{buf_name}.get(), NULL));"""
} else {
    // 如果输出参数是张量元组，则逐个处理每个张量
    for idx, output_arg in enumerate(output_args):
        lines += f"""
{output_arg} =
    reinterpret_cast<AtenTensorHandle>(PyCapsule_GetPointer(PyList_GET_ITEM(py_{buf_name}.get(), {idx}), NULL));"""
}

// 为输出参数声明 RAIIAtenTensorHandle 变量
declarations_before_scope = [
    f"RAIIAtenTensorHandle {output_arg};"
    for idx, output_arg in enumerate(output_args)
]
// 生成获取全局解释器锁的作用域
scope_gil_acquire = self.generate_scoped_gil_acquire(
    declarations_before_scope, lines
)
// 将作用域锁代码写入输出
self.writelines(scope_gil_acquire)
    # 根据输入的类型返回对应的 C 类型字符串
    def c_type_for_prim_type(self, type_) -> str:
        # 断言在 ABI 兼容模式下进行，否则抛出异常
        assert (
            config.abi_compatible
        ), "c_type_for_prim_type is only used in ABI compatible mode"
        
        # 如果类型是 OptionalType，则返回其元素类型加上指针符号
        if isinstance(type_, torch.OptionalType):
            return f"{self.c_type_for_prim_type(type_.getElementType())}*"
        # 如果类型是 TensorType，则返回 AtenTensorHandle
        elif isinstance(type_, torch.TensorType):
            return "AtenTensorHandle"
        # 如果类型是 IntType 或 SymIntType，则返回 int64_t
        elif isinstance(type_, (torch.IntType, torch.SymIntType)):
            return "int64_t"
        # 如果类型是 BoolType、SymBoolType、EnumType 或者是 ScalarType 或 Layout 的字符串表示，则返回 int32_t
        elif isinstance(
            type_, (torch.BoolType, torch.SymBoolType, torch.EnumType)
        ) or repr(type_) in ("ScalarType", "Layout"):
            return "int32_t"
        # 如果类型是 FloatType，则返回 double
        elif isinstance(type_, torch.FloatType):
            return "double"
        # 否则，抛出异常，表明遇到了意外的类型
        else:
            raise AssertionError(f"Unexpected type in c_type_for_prim_type: {type_=}")

    # 根据输入的值和类型返回对应的参数字符串表示
    def val_to_arg_str_for_prim_type(self, val, type_) -> str:
        # TODO: not using type_ as the first step of refactoring. Will update this later.
        
        # 如果值是布尔类型
        if isinstance(val, bool):
            # 如果在 ABI 兼容模式下，返回 "1" 或 "0"
            if config.abi_compatible:
                return "1" if val else "0"
            # 否则，返回 "true" 或 "false"
            else:
                return "true" if val else "false"
        
        # 如果值是整数类型
        elif isinstance(val, int):
            # 如果运行平台是 Darwin（MacOS），返回值后加上 "LL"
            if sys.platform == "darwin":
                return f"{val}LL"
            # 否则，返回值后加上 "L"
            else:
                return f"{val}L"
        
        # 如果值是字符串类型
        elif isinstance(val, str):
            return f'"{val}"'
        
        # 如果值是特定类型（Buffer、ReinterpretView、StorageBox、TensorBox），返回其代码生成的引用
        elif isinstance(
            val, (ir.Buffer, ir.ReinterpretView, ir.StorageBox, ir.TensorBox)
        ):
            return val.codegen_reference()
        
        # 如果值是 torch.device 类型，返回其代码生成的设备字符串表示
        elif isinstance(val, torch.device):
            return self.codegen_device(val)
        
        # 如果值是 torch.dtype 类型，返回其代码生成的数据类型字符串表示
        elif isinstance(val, torch.dtype):
            return self.codegen_dtype(val)
        
        # 如果值是浮点数类型且为正无穷或负无穷，返回对应的 C++ 标准库表示
        elif isinstance(val, float) and val in [float("inf"), float("-inf")]:
            if val == float("inf"):
                return "std::numeric_limits<float>::infinity()"
            else:
                return "-std::numeric_limits<float>::infinity()"
        
        # 如果值是列表或元组类型
        elif isinstance(val, (list, tuple)):
            # FIXME: 这是因为 type_ 并不总是正确设置为 torch.ListType 导致的问题
            # 返回值列表中每个元素转换为参数字符串后的集合表示
            return f"{{{', '.join(self.val_to_arg_str(x, None) for x in val)}}}"
        
        # 否则，返回值的 repr() 字符串表示
        else:
            return repr(val)
```