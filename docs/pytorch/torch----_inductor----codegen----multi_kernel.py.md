# `.\pytorch\torch\_inductor\codegen\multi_kernel.py`

```py
# mypy: allow-untyped-defs
# 引入日志模块
import logging
# 引入操作系统接口模块
import os
# 引入类型提示模块中的特定类型
from typing import Any, List

# 引入 Torch 框架的指标表和指标表启用状态检查函数
from torch._inductor.metrics import get_metric_table, is_metric_table_enabled

# 引入上级目录中的配置模块
from .. import config
# 引入代码缓存模块和 Triton 未来模块
from ..codecache import PyCodeCache, TritonFuture
# 引入运行时工具中的 GPU 性能测试函数
from ..runtime.runtime_utils import do_bench_gpu
# 引入缓存装饰器
from ..utils import cache_on_self
# 引入虚拟化模块 V
from ..virtualized import V
# 引入通用的张量参数模块
from .common import TensorArg

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)


def get_kernel_argdefs(kernel):
    # 调用 kernel 对象的 python_argdefs 方法获取其参数定义
    arg_defs, _, _, _ = kernel.args.python_argdefs()
    return arg_defs


def _get_all_args(args_list, arg_types_list=None):
    # 找出参数列表中最长的参数列表
    all_args = max(args_list, key=len)[:]
    # 如果提供了参数类型列表，也找出其中最长的参数类型列表
    arg_types = max(arg_types_list, key=len)[:] if arg_types_list is not None else None
    # 确保每个参数列表是最长参数列表的子集
    for args in args_list:
        assert set(args).issubset(set(all_args)), f"{args} v.s. {all_args}"

    return all_args, arg_types


def get_all_kernel_argdefs(kernels):
    """
    根据一组 kernels，获取所有 kernel 的参数定义。

    此处的逻辑必须与 `get_all_call_args` 相匹配，但此处不需要获取参数类型。
    """
    # 获取每个 kernel 的参数定义并组成列表
    argdefs_list = [get_kernel_argdefs(kernel) for kernel in kernels]

    # 调用 _get_all_args 函数，返回所有 kernel 的参数定义组成的列表
    return _get_all_args(argdefs_list)[0]


def get_all_call_args(call_args_list, arg_types_list):
    """
    传入每个子 kernel 的调用参数列表，并返回组合的多 kernel 的调用参数。

    注意，以下算法并非总是有效：
    ```
        all_call_args: Dict[
            Any, None
        ] = {}  # use a dict rather than set to maintain insertion order
        for call_args in call_args_list:
            all_call_args.update({arg: None for arg in call_args})

        all_call_args = list(all_call_args.keys())
    ```py
    如果任何 kernel 多次传入相同参数，此算法将失败。
    请参见 test_pass_same_arg_multi_times 中的测试用例 test_multi_kernel.py

    相反，我们选择最长的调用参数列表，并断言其他调用参数是其子集。
    """
    # 调用 _get_all_args 函数，返回所有调用参数列表的最长列表和参数类型列表
    return _get_all_args(call_args_list, arg_types_list)


def get_numel_argdefs(kernel):
    # 初始化 numel 参数定义列表
    numel_argdefs = []
    # 遍历 kernel 的 range_trees
    for tree in kernel.range_trees:
        # 如果树的前缀不是 "r" 或 kernel 内部为 reduction
        if tree.prefix != "r" or kernel.inside_reduction:
            # 将参数定义添加到 numel_argdefs 中
            numel_argdefs.append(f"{tree.prefix}numel")

    return numel_argdefs


class MultiKernelState:
    """
    维护多 kernel 编译状态，以防止为相同子 kernel 集合定义重复的多 kernel。

    V.graph.wrapper_code 引用了 MultiKernelState 实例。
    """

    def __init__(self):
        # 子 kernel 到 kernel 名称的映射字典
        self.subkernel_to_kernel_name = {}
        def define_kernel(self, kernels):
            """
            Previously we name the multi kernel as "multi_kernel_{kernel_names[0]}".
            This has some minor issue.

            E.g. for persistent reduction https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca ,
            there are 2 flavors of non-persistent reduction:
              https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4
            and
              https://gist.github.com/shunting314/02ee753b65c513c54e695626afe682bd

            The only different is cache eviction policy.

            We should name the multi-kernel differently in these 2 cases.
            """
            # Collecting kernel names from input kernel objects
            kernel_names = tuple(k.kernel_name for k in kernels)
            # Check if the kernel_names tuple is already mapped in subkernel_to_kernel_name
            if kernel_names in self.subkernel_to_kernel_name:
                return self.subkernel_to_kernel_name[kernel_names]

            # name the multi kernel based on the number of entries in subkernel_to_kernel_name
            multi_kernel_name = f"multi_kernel_{len(self.subkernel_to_kernel_name)}"
            # Assign the newly created multi_kernel_name to the current kernel_names
            self.subkernel_to_kernel_name[kernel_names] = multi_kernel_name

            # Check if cpp_wrapper flag is set; if yes, return multi_kernel_name
            if V.graph.cpp_wrapper:
                return multi_kernel_name

            # Obtain wrapper code from V.graph
            wrapper = V.graph.wrapper_code

            # Generate kernel call definitions for each kernel in kernels
            kernel_call_def_code = "\n".join(
                [
                    f"""
        def call{idx}(need_clone_args=False):
            args = [{', '.join(get_kernel_argdefs(kernels[idx]))}]
            if need_clone_args:
                args, _ = multi_kernel_call.kernels[{idx}].clone_args(*args)
            multi_kernel_call.kernels[{idx}].run(*args, {', '.join(get_numel_argdefs(kernels[idx]))}, grid=grid, stream=stream)
                    """.format(
                        idx
                    ).strip(
                        "\n"
                    )
                    for idx in range(len(kernels))
                ]
            )

            # Add subkernel source code hashes to the multi-kernel source code for cache handling
            subkernel_hashes = "\n".join(
                f"# subkernel{i} code hash: {kernel.code_hash}"
                for i, kernel in enumerate(kernels)
            )

            src_code = f"""
{subkernel_hashes}
def run(multi_kernel_call, {', '.join(get_all_kernel_argdefs(kernels))}, {', '.join(get_numel_argdefs(kernels[0]))}, grid, stream):
    """
    Define a function `run` that executes a multi-kernel call.

    Args:
        multi_kernel_call: The multi-kernel call object to execute.
        <get_all_kernel_argdefs(kernels)>: Arguments specific to all kernels.
        <get_numel_argdefs(kernels[0])>: Arguments related to the first kernel's size.
        grid: Grid configuration for the kernels.
        stream: Stream configuration for execution.

    Returns:
        multi_kernel_name: The name of the compiled multi-kernel.

    Note:
        This function generates and compiles a multi-kernel call based on provided kernels
        and their configurations.
    """
    multi_kernel_compile = f"""
{multi_kernel_name} = async_compile.multi_kernel({multi_kernel_name!r}, [
    {", ".join(kernel_names)},
],
'''
{src_code}
'''
)"""
    """
    Generate and compile a multi-kernel using asynchronous compilation.

    Args:
        multi_kernel_name: Name of the multi-kernel being compiled.
        kernel_names: List of names of individual kernels to include in the multi-kernel.
        src_code: Source code defining the multi-kernel.

    Note:
        This block constructs and compiles a multi-kernel using async compilation,
        incorporating specified kernels and source code.
    """

    wrapper.header.splice(multi_kernel_compile)
    """
    Integrate the compiled multi-kernel definition into the wrapper header.

    Args:
        multi_kernel_compile: Compiled definition of the multi-kernel.

    Note:
        This step inserts the compiled multi-kernel definition into the wrapper header
        for further processing or inclusion in generated code.
    """

    if config.triton.autotune_at_compile_time:
        wrapper.kernel_autotune_defs.splice(multi_kernel_compile)
    """
    Optionally splice the multi-kernel compile information for autotuning.

    Args:
        multi_kernel_compile: Compiled definition of the multi-kernel.

    Note:
        This conditionally adds the multi-kernel compile information to the wrapper
        for autotuning purposes, based on configuration settings.
    """

    return multi_kernel_name
    """
    Return the name of the compiled multi-kernel.

    Returns:
        multi_kernel_name: Name of the compiled multi-kernel.

    Note:
        This function returns the name of the compiled multi-kernel for subsequent use
        or reference.
    """


class MultiKernel:
    """
    This class maintains the compile time state for multi kernels.

    Assume we do codegen for a MultiKernel encapsulating kernel1 and kernel2.
    The generated definition for the multi-kernel will looks like:
    ```
    multi_kernel_kernel1 = MultiKernelCall([kernel1, kernel2], multi_kernel_definition_code)
    ```py

    Here is an concrete example: https://gist.github.com/shunting314/d9f3fb6bc6cee3dbae005825ca196d39
    """

    def __init__(self, kernels):
        """
        Initialize a MultiKernel object.

        Args:
            kernels: List of kernels to encapsulate within the MultiKernel.

        Raises:
            AssertionError: If fewer than 2 kernels are provided.

        Note:
            This constructor initializes the MultiKernel object with a list of kernels
            and defines the kernel name in the wrapper code for state management.
        """
        assert len(kernels) >= 2

        self.kernels = kernels
        self.kernel_name = V.graph.wrapper_code.multi_kernel_state.define_kernel(
            kernels
        )
        """
        Define the kernel name for the MultiKernel in the wrapper code.

        Args:
            kernels: List of kernels encapsulated within the MultiKernel.

        Note:
            This step defines the kernel name in the wrapper code to maintain state
            related to the MultiKernel's compilation and execution.
        """

        # need this since some code in inductor check if the kernel object has an args
        # attribute to decide if it's a non-null kernel.
        self.args = object()
        """
        Create a placeholder `args` attribute.

        Note:
        This assigns a placeholder `args` attribute to the MultiKernel object,
        ensuring compatibility with certain checks in the inductor code to identify
        valid kernel objects.
        """
    def call_kernel(self, kernel_name):
        """
        Collect the union of arguments from all subkernels as the arguments
        for the multi-kernel.
        """
        # 确保传入的 kernel_name 与实例中保存的 kernel_name 一致
        assert kernel_name == self.kernel_name
        # 调用图包装代码中的 Triton 头部写入函数
        V.graph.wrapper_code.write_triton_header_once()
        # 初始化空的调用参数列表和参数类型列表
        call_args_list = []
        arg_types_list = []
        # 遍历所有子内核
        for kernel in self.kernels:
            # 调用子内核对象的 python_argdefs 方法，获取返回的参数元组
            _, call_args, _, arg_types = kernel.args.python_argdefs()
            # 将子内核的调用参数和参数类型添加到列表中
            call_args_list.append(call_args)
            arg_types_list.append(arg_types)

        # 调用函数，获取所有子内核的调用参数和参数类型的并集
        all_call_args, arg_types = get_all_call_args(call_args_list, arg_types_list)
        # 初始化网格为一个空列表
        grid: List[Any] = []

        # 如果启用了 cpp-wrapper，则选取特定的内核进行快速调用
        if V.graph.cpp_wrapper:
            # 查找并选择特定内核的索引
            picked_kernel = MultiKernelCall.lookup_choice(kernel_name)
            # 更新 kernel_name 为选择的内核的 kernel_name
            kernel_name = self.kernels[picked_kernel].kernel_name
            # 更新 final_call_args 和 arg_types 为选择的内核的参数和参数类型
            final_call_args = call_args_list[picked_kernel]
            arg_types = arg_types_list[picked_kernel]
        else:
            # 否则，使用所有子内核的联合调用参数作为 final_call_args
            final_call_args = all_call_args

        # 将第一个子内核的 numel 添加到调用参数和网格中
        self.kernels[0].add_numel_to_call_args_and_grid(
            kernel_name, final_call_args, arg_types, grid
        )

        # 生成默认网格并更新到 grid 变量中
        grid = V.graph.wrapper_code.generate_default_grid(kernel_name, grid)
        # 生成内核调用的代码
        V.graph.wrapper_code.generate_kernel_call(
            kernel_name,
            final_call_args,
            grid,
            arg_types=arg_types,
        )

    def codegen_nan_check(self):
        """
        Generate assertions for NaN and infinity checks in the generated code.
        """
        # 获取图包装代码对象
        wrapper = V.graph.wrapper_code
        # 初始化一个空集合用于跟踪已见过的参数
        seen = set()
        # 遍历所有子内核
        for k in self.kernels:
            # 调用子内核对象的 python_argdefs 方法，获取返回的参数元组
            _, call_args, precompile_args, _ = k.args.python_argdefs()
            # 遍历每个参数和预编译参数的元组
            for arg, precompile_arg in zip(call_args, precompile_args):
                # 如果参数已经在 seen 集合中，则跳过
                if arg in seen:
                    continue
                # 将参数添加到 seen 集合中
                seen.add(arg)
                # 如果预编译参数是 TensorArg 类型
                if isinstance(precompile_arg, TensorArg):
                    # 生成断言，确保参数中没有 NaN 值
                    line = f"assert not {arg}.isnan().any().item()"
                    wrapper.writeline(line)
                    # 生成断言，确保参数中没有 Infinity 值
                    line = f"assert not {arg}.isinf().any().item()"
                    wrapper.writeline(line)

    @property
    def removed_buffers(self):
        """
        Return the intersection of removed buffers across all kernels.
        """
        return set.intersection(*[k.removed_buffers for k in self.kernels])

    @property
    def inplaced_to_remove(self):
        """
        Return the intersection of inplaced buffers to remove across all kernels.
        """
        return set.intersection(*[k.inplaced_to_remove for k in self.kernels])

    @property
    @cache_on_self
    def inplace_update_buffers(self):
        """
        Ensure all kernels have the same inplace update mappings.
        """
        # 对所有子内核检查是否具有相同的 in-place 更新映射
        for k in self.kernels[1:]:
            assert k.inplace_update_buffers == self.kernels[0].inplace_update_buffers
        # 返回第一个子内核的 in-place 更新映射
        return self.kernels[0].inplace_update_buffers

    def warn_mix_layout(self, kernel_name: str):
        """
        Placeholder function to warn about mixed tensor layouts.
        """
        pass
class MultiKernelCall:
    """
    This class is called at run time to actually run the kernel
    """

    def __init__(self, multi_kernel_name, kernels, src_code):
        # 确保 kernels 至少包含两个元素
        assert len(kernels) >= 2
        self._kernels = kernels
        self.multi_kernel_name = multi_kernel_name

        # 加载源代码并获取其运行函数
        self._run = PyCodeCache.load(src_code).run
        # 检查是否禁用多内核缓存
        self.disable_cache = os.environ.get(
            "TORCHINDUCTOR_DISABLE_MULTI_KERNEL_CACHE"
        ) == "1" or is_metric_table_enabled("persistent_red_perf")

        self.picked_kernel = None
        if config.triton.multi_kernel > 1:
            # 根据配置手动选择子内核以便进行性能测试
            picked_by_config = config.triton.multi_kernel - 2
            assert picked_by_config < len(self._kernels)
            self.picked_kernel = picked_by_config
        elif not self.disable_cache:
            self.load_cache()

        self._recorded = False

    def cache_file_path(self):
        # 获取缓存文件路径
        py_file_path = self._run.__globals__["__file__"]
        return os.path.splitext(py_file_path)[0] + ".picked_kernel"

    def load_cache(self):
        # 加载缓存中的选择的内核
        assert self.picked_kernel is None
        path = self.cache_file_path()
        if os.path.exists(path):
            with open(path) as fd:
                self.picked_kernel = int(fd.read())
                assert self.picked_kernel >= 0 and self.picked_kernel < len(
                    self._kernels
                )
                log.debug(
                    "Load picked kernel %d from cache file %s", self.picked_kernel, path
                )

    def store_cache(self):
        # 存储选择的内核到缓存文件
        assert self.picked_kernel is not None
        path = self.cache_file_path()
        with open(path, "w") as fd:
            fd.write(str(self.picked_kernel))
        log.debug("Store picked kernel %d to cache file %s", self.picked_kernel, path)

    @property
    def kernels(self):
        """
        Read results from future.

        This should be called after parallel compilation is done.
        In case you call this before compilation is done,
        it may slow down the parallel compilation.
        """
        # 从所有异步编译后的内核中读取结果
        for i, kernel in enumerate(self._kernels):
            if isinstance(kernel, TritonFuture):
                self._kernels[i] = kernel.result()

        return self._kernels

    def run(self, *args, **kwargs):
        # 执行运行函数
        self._run(self, *args, **kwargs)

    @staticmethod
    def benchmark_sub_kernels(kernel_calls):
        """
        Benchmark all the sub kernels and return the execution time
        (in milliseconds) for each of time.

        Unit test may mock this method to force a specific kernel to
        be picked.
        """
        # 对所有子内核进行基准测试，并返回每个的执行时间（毫秒）
        return [
            do_bench_gpu(lambda: kernel_call(True), rep=40, fast_flush=True)
            for kernel_call in kernel_calls
        ]

    # record_choice and lookup_choice are helper functions for cpp-wrapper
    # codegen. The first pass use record_choice to keep the choice and
    # 在第二次通过调用 lookup_choice 进行查找。
    #
    # 重复使用多核缓存的替代方案效果不佳，
    # 因为在第二次通过的代码生成期间，很难知道缓存文件的路径。
    # 同时，读取缓存文件需要进行一些IO操作，可能会更慢。
    @staticmethod
    def record_choice(multi_kernel_name, choice):
        """
        记录多核选择以用于第二次通过的cpp-wrapper代码生成。

        如果在代码生成期间未调用此函数，则不执行任何操作。
        """
        from torch._inductor.graph import GraphLowering

        if not isinstance(V.graph, GraphLowering):
            return

        if not V.graph.record_multi_kernel_choice:
            return

        V.graph.multi_kernel_to_choice[multi_kernel_name] = choice

    @staticmethod
    def lookup_choice(multi_kernel_name):
        # 这应该始终在cpp-wrapper代码生成期间完成
        assert V.graph.record_multi_kernel_choice
        # 应该不会找不到
        return V.graph.multi_kernel_to_choice[multi_kernel_name]

    def run_with_argless_kernels(self, kernel_calls):
        if self.picked_kernel is None:
            timings = self.benchmark_sub_kernels(kernel_calls)
            self.picked_kernel = timings.index(min(timings))
            k0 = self.kernels[0]
            log.debug(
                "pick %dth sub-kernel in %s. Size hints %s. Reduction hint %s. Timings %s",
                self.picked_kernel,
                [k.inductor_meta.get("kernel_name") for k in self.kernels],
                k0.size_hints,
                k0.inductor_meta.get("reduction_hint"),
                timings,
            )

            def get_kernel_path(k):
                return k.fn.fn.__code__.co_filename

            get_metric_table("persistent_red_perf").add_row(
                lambda: {
                    "kernel1_name": get_kernel_path(self.kernels[0]),
                    "kernel2_name": get_kernel_path(self.kernels[1]),
                    "kernel1_latency": timings[0],
                    "kernel2_latency": timings[1],
                    "size_hints": k0.size_hints,
                    "reduction_hint": k0.inductor_meta.get("reduction_hint"),
                    "speedup": timings[1] / timings[0],
                }
            )

            if not self.disable_cache:
                self.store_cache()

        if not self._recorded:
            self._recorded = True
            self.record_choice(self.multi_kernel_name, self.picked_kernel)
        kernel_calls[self.picked_kernel]()
```