# `.\pytorch\test\test_dispatch.py`

```
# Owner(s): ["module: dispatch"]

# 导入必要的模块和库
import itertools  # 提供迭代工具
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作

from collections import namedtuple  # 导入命名元组的类

import torch._C as C  # 导入torch的C扩展模块
import torch.utils.cpp_extension  # 导入用于C++扩展的工具
from torch._python_dispatcher import PythonDispatcher  # 导入Python调度器
from torch.testing._internal.common_utils import run_tests, TestCase  # 导入测试运行和测试用例类

# TODO: 扩展调度器 API 以成为一个通用的接口，用于从 Python 与调度器交互！
#
# 这些是用于调度行为的交换性的详尽测试。如果您需要更多的使用信息样式测试，请查看op_registration_test.cpp
#
# 这里没有测试的内容：
#   - 监听器
#   - 顶层命名空间注册
#   - 回退（fallback）
#   - CppFunction/schema 的特殊重载
#
# 这里不直接测试的内容：
#   - 调度器的内部状态是否合理。这间接通过不变量测试来验证

Result = namedtuple("Result", "state table provenance")

# 需要检查的调度键列表
dispatch_keys_to_check = (
    "Undefined",
    "CPU",
    "CUDA",
    "XLA",
    "AutogradOther",
    "AutogradCPU",
    "AutogradCUDA",
    "AutogradXLA",
)


def extract_dispatch_table_with_keys(table, dispatch_keys):
    extracted = ""  # 初始化一个空字符串，用于保存提取的结果
    table_entries = table.split("\n")  # 将表格按行分割成条目列表
    regex = re.compile(r"registered at .*FallbackKernel\.cpp.*(\[)")  # 编译正则表达式，用于匹配特定格式
    for k in dispatch_keys:  # 遍历每个调度键
        for t in table_entries:  # 遍历每个表格条目
            if t.startswith(k):  # 如果条目以当前调度键开头
                # 屏蔽掉树内后备机制的文件:行信息
                entry = regex.sub("registered in pytorch framework [", t)
                extracted += entry + "\n"  # 将处理后的条目添加到提取结果中
    return extracted  # 返回提取的条目字符串


class TestDispatch(TestCase):
    namespace_index = 0  # 定义测试类的命名空间索引

    def test_all_invariants(self):
        # 检查调度器的所有不变量是否满足
        C._dispatch_check_all_invariants()

    # 您可能不希望直接调用此方法；如果您的构造函数不满足交换性，则仍然可以使用固定的构造器顺序运行commute，
    # 以便测试析构函数是否仍然满足交换性
    def run_ops(
        self,
        name,
        ops,
        ctor_order=None,
        dtor_order=None,
        results=None,
        expect_raises=False,
    ):
        # 操作符注册是可交换的（因为静态初始化器可以以任何顺序运行），并且可逆（通过取消注册）。 （受一些注意事项的限制：系统中的一些遗留行为不是可交换的-我们希望消除这些行为！）
        #
        # 因此，虽然原则上我们可以简单地按照用户指定的顺序逐个运行一组操作，
        # 但通过做更多的工作，我们可以更加确保这些额外属性：
        #
        # 1. 不仅按固定顺序运行注册：运行每个可能的排列。同样，运行取消注册的每个排列。
        #
        # 2. 不仅仅检查调度器的最终状态：对于每个操作符注册的子集，确保计算的中间状态是路径无关的。需要注意的一点：
        #
        # ```
        pass  # 这是一个空方法，用于暂时保持结构完整，没有具体实现
    # 定义一个函数用于测试操作的可交换性，参数包括操作的名称、操作列表、构造函数顺序、是否期望引发异常
    # 结果将保存在 results 字典中
    def commute(self, name, ops, ctor_order=None, expect_raises=False):
        results = {}

        # 定义内部函数 go，用于处理给定的构造函数顺序 ctor_order 和析构函数顺序 dtor_order 的所有排列
        def go(ctor_order):
            # 遍历 ops 列表的所有排列
            for dtor_order in itertools.permutations(range(len(ops))):
                # 调用 run_ops 方法执行操作
                self.run_ops(
                    name,
                    ops,
                    ctor_order,
                    dtor_order,
                    results=results,
                    expect_raises=expect_raises,
                )

        # 如果指定了构造函数顺序 ctor_order，则只处理这个顺序
        if ctor_order is not None:
            go(ctor_order)
        else:
            # 否则遍历 ops 列表的所有排列的构造函数顺序
            for ctor_order in itertools.permutations(range(len(ops))):
                go(ctor_order)

        # 返回运行完所有操作后的 results 字典中完整的 Result 命名元组
        # 如果 KeyError，表示无法找到任何构造函数的顺序使得所有操作完成，这是测试构造上的错误
        return results[frozenset(range(len(ops)))]

    # 定义一个测试函数 test_def，测试 def 操作的多种情况
    def test_def(self):
        # 调用 commute 方法进行测试，获取状态 state
        state = self.commute(
            "foo",
            [
                # 第一种操作：定义 foo 函数接受 Tensor 类型参数并返回 Tensor 类型
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 第二种操作：实现 foo 函数，无特定分发方式
                lambda m: m.impl_t_t("foo"),
                # 第三种操作：实现 foo 函数，指定为在 CPU 上执行
                lambda m: m.impl_t_t("foo", dispatch="CPU"),
                # 第四种操作：实现 foo 函数，指定为在 Autograd 模式下执行
                lambda m: m.impl_t_t("foo", dispatch="Autograd"),
                # 第五种操作：实现 foo 函数，指定为在 AutogradCPU 模式下执行
                lambda m: m.impl_t_t("foo", dispatch="AutogradCPU"),
            ],
        ).state
        
        # 断言测试状态符合预期
        self.assertExpectedInline(
            state,
            """\
    def test_def_impl_schema_mismatch(self):
        # NB: an impl-impl mismatch is not reported eagerly; you'll find out
        # about it because one of them won't match with def
        # 定义一个测试用例，验证当实现与定义不匹配时，是否会报错
        state = self.commute(
            "foo",
            [
                # m.def("foo(Tensor x, Tensor y) -> Tensor")
                # 使用 lambda 函数定义一个接受两个 Tensor 参数并返回 Tensor 的函数
                lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
                # m.impl("foo", [](const Tensor & x) { return x })
                # 使用 lambda 函数实现一个接受 Tensor 参数并返回 Tensor 的函数
                lambda m: m.impl_t_t("foo"),
            ],
            expect_raises=True,  # 期望这里会引发异常
        ).state
        # 断言期望的输出结果
        self.assertExpectedInline(
            state,
            """\
Inferred operator schema for a C++ kernel function doesn't match the expected function schema.
  operator: test::foo
  expected schema: test::foo(Tensor x, Tensor y) -> Tensor
    registered at /dev/null:0
  inferred schema: (Tensor _0) -> Tensor _0
    impl_t_t
  reason: The number of arguments is different. 2 vs 1.""",
        )

    def test_def_with_inference(self):
        state = self.commute(
            "foo",
            [
                # m.def("foo", [](const Tensor & x) { return x })
                # 使用 lambda 函数定义一个接受一个 Tensor 参数并返回 Tensor 的函数
                lambda m: m.def_name_t_t("foo"),
                # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
                # 使用 lambda 函数实现一个在 CPU 上运行的接受 Tensor 参数并返回 Tensor 的函数
                lambda m: m.impl_t_t("foo", "CPU"),
                # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
                # 使用 lambda 函数实现一个在 Autograd 上运行的接受 Tensor 参数并返回 Tensor 的函数
                lambda m: m.impl_t_t("foo", "Autograd"),
                # m.impl("foo", torch::kAutogradCPU, [](const Tensor & x) { return x })
                # 使用 lambda 函数实现一个在 AutogradCPU 上运行的接受 Tensor 参数并返回 Tensor 的函数
                lambda m: m.impl_t_t("foo", "AutogradCPU"),
            ],
        ).state
        # 断言期望的输出结果
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor _0) -> Tensor _0
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
AutogradCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
Autograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

    def test_def_only(self):
        state = self.commute(
            "foo",
            [
                # m.def("foo(Tensor x, Tensor y) -> Tensor")
                # 使用 lambda 函数定义一个接受两个 Tensor 参数并返回 Tensor 的函数
                lambda m: m.def_("foo(Tensor x, Tensor y) -> Tensor"),
            ],
        ).state
        # 断言期望的输出结果
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> Tensor
debug: registered at /dev/null:0
""",
        )
    def test_impl_only(self):
        # 设定一个初始状态，使用 self.commute 方法
        state = self.commute(
            "foo",
            [
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数
                lambda m: m.impl_t_t("foo"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 CPU 上的实现
                lambda m: m.impl_t_t("foo", "CPU"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 Autograd 模式下的实现
                lambda m: m.impl_t_t("foo", "Autograd"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 AutogradCPU 模式下的实现
                lambda m: m.impl_t_t("foo", "AutogradCPU"),
            ],
        ).state
        # 断言预期结果，检查状态的输出格式
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: (none)
CPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
AutogradCPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
Autograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

    def test_computed_table(self):
        # 调用 self.commute 方法生成一个状态和表格结果
        result = self.commute(
            "foo",
            [
                # 定义一个 lambda 函数，调用 m.def_name_t_t 方法定义 "foo" 函数
                lambda m: m.def_name_t_t("foo"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 CPU 上的实现，并且指定 debug 为 "fn_cpu"
                lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 XLA 上的实现，并且指定 debug 为 "fn_xla"
                lambda m: m.impl_t_t("foo", "XLA", debug="fn_xla"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 Autograd 模式下的实现，并且指定 debug 为 "fn_autograd"
                lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
                # 定义一个 lambda 函数，调用 m.impl_t_t 方法实现 "foo" 函数，指定在 AutogradCPU 模式下的实现，并且指定 debug 为 "fn_autogradcpu"
                lambda m: m.impl_t_t("foo", "AutogradCPU", debug="fn_autogradcpu"),
            ],
        )
        # 从结果中获取状态和表格
        state, table = result.state, result.table
        # 断言预期结果，检查状态的输出格式
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor _0) -> Tensor _0
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
XLA: fn_xla :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
AutogradCPU: fn_autogradcpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 由于计算的分派表格过大，只检查我们感兴趣的几个条目
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check
        )

        # 断言预期结果，检查抽取的表格内容
        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: default_def_name_t_t [math kernel]
CPU: fn_cpu [kernel]
""",
        )
    def test_computed_table_with_cpu_math_autogradcpu_fallthrough(self):
        # 获取全局调度库对象，指定库名为 "IMPL"，后缀为 "_"，处理器类型为 "AutogradCPU"
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        # 进行函数调度测试，并返回结果对象
        result = self.commute(
            "foo",
            [
                # 定义名为 "foo" 的函数，接受一个 Tensor 参数并返回 Tensor
                lambda m: m.def_name_t_t("foo"),
                # 将名为 "foo" 的函数实现为在 CPU 上执行的版本，接受一个 Tensor 参数并返回 Tensor
                lambda m: m.impl_t_t("foo", "CPU"),
            ],
        )
        # 获取结果状态和调度表
        state, table = result.state, result.table
        # 断言期望结果与内联字符串匹配
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor _0) -> Tensor _0
debug: registered at /dev/null:0
alias analysis kind: CONSERVATIVE
CPU: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: default_def_name_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 提取关键字对应的调度表条目，并赋值给 extracted_table
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check
        )

        # 断言提取的调度表与内联字符串匹配
        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: default_def_name_t_t [math kernel]
CPU: impl_t_t [kernel]
CUDA: default_def_name_t_t [math kernel]
XLA: default_def_name_t_t [math kernel]
AutogradOther: default_def_name_t_t [math kernel]
AutogradCPU: registered in pytorch framework [backend fallback]
AutogradCUDA: default_def_name_t_t [math kernel]
AutogradXLA: default_def_name_t_t [math kernel]
""",
        )

    def test_computed_table_with_math(self):
        # 获取全局调度库对象，指定库名为 "IMPL"，后缀为 "_"，处理器类型为 "AutogradCPU"
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        # 进行函数调度测试，并返回结果对象
        result = self.commute(
            "foo",
            [
                # 定义名为 "foo(Tensor x) -> Tensor" 的函数
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 将名为 "foo" 的函数实现为复合隐式自动微分的版本，接受一个 Tensor 参数并返回 Tensor
                lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd"),
            ],
        )
        # 获取结果状态和调度表
        state, table = result.state, result.table
        # 断言期望结果与内联字符串匹配
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CompositeImplicitAutograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 提取关键字对应的调度表条目，并赋值给 extracted_table
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check
        )

        # 断言提取的调度表与内联字符串匹配
        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: impl_t_t [math kernel]
    def test_computed_table_with_cpu_math(self):
        # 获取 AutogradCPU 实现的全局变量
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        # 进行调度表的计算，并获取结果
        result = self.commute(
            "foo",
            [
                # 定义 foo 函数的签名
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 使用 CPU 实现定义 foo 函数
                lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
                # 使用 CompositeImplicitAutograd 实现定义 foo 函数
                lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
            ],
        )
        # 提取结果状态和调度表
        state, table = result.state, result.table
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 由于计算得到的调度表过大，只检查我们感兴趣的几个条目。
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check
        )

        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: fn_math [math kernel]
CPU: fn_cpu [kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: fn_math [math kernel]
AutogradCPU: registered in pytorch framework [backend fallback]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
""",
        )

    def test_computed_table_with_autograd(self):
        # 获取 AutogradCPU 实现的全局变量
        global_m = C._dispatch_library("IMPL", "_", "AutogradCPU")
        # 进行调度表的计算，并获取结果
        result = self.commute(
            "foo",
            [
                # 定义 foo 函数的签名
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 使用 Autograd 实现定义 foo 函数
                lambda m: m.impl_t_t("foo", "Autograd"),
            ],
        )
        # 提取结果状态和调度表
        state, table = result.state, result.table
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
Autograd[alias]: impl_t_t :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )
    # computed dispatch table is too big, so we only check on a few entries we're interested in.
    # 计算得到的分发表过大，因此我们只检查我们感兴趣的几个条目。

    extracted_table = extract_dispatch_table_with_keys(
        table, dispatch_keys_to_check
    )
    # 使用指定的键从表中提取分发表

    self.assertExpectedInline(
        extracted_table,
        """\
AutogradOther: impl_t_t [autograd kernel]
AutogradCPU: impl_t_t [autograd kernel]
AutogradCUDA: impl_t_t [autograd kernel]
AutogradXLA: impl_t_t [autograd kernel]
""",
    )
    # 断言提取出的分发表内容与预期的内容匹配

# Now that catchAll maps to CompositeImplicitAutograd, registering to both
# catchAll and CompositeImplicitAutograd breaks commutativity.
# 现在 catchAll 映射到 CompositeImplicitAutograd，同时注册到 catchAll 和 CompositeImplicitAutograd 会破坏可交换性。
def test_computed_table_with_cpu_autograd_math(self):
    # 进行操作并获取结果
    result = self.commute(
        "foo",
        [
            # m.def("foo(Tensor x) -> Tensor")
            lambda m: m.def_("foo(Tensor x) -> Tensor"),
            # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
            # m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
            # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
            lambda m: m.impl_t_t(
                "foo", "CompositeImplicitAutograd", debug="fn_math"
            ),
        ],
    )
    # 获取结果的状态和表格
    state, table = result.state, result.table

    self.assertExpectedInline(
        state,
        """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
    )
    # 断言状态内容与预期的内容匹配

    # computed dispatch table is too big, so we only check on a few entries we're interested in.
    # 计算得到的分发表过大，因此我们只检查我们感兴趣的几个条目。
    extracted_table = extract_dispatch_table_with_keys(
        table, dispatch_keys_to_check
    )

    self.assertExpectedInline(
        extracted_table,
        """\
Undefined: fn_math [math kernel]
CPU: fn_cpu [kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: fn_math [math kernel]
AutogradCPU: fn_autograd [autograd kernel]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
""",
    )
    # 断言提取出的分发表内容与预期的内容匹配
    def test_computed_table_with_ambiguous_autogradother(self):
        # 调用测试函数，验证处理带有模糊性的自动微分和其它操作的情况
        result = self.commute(
            "foo",
            [
                # 定义名为 "foo" 的函数接口，接受参数为 Tensor 类型并返回 Tensor 类型
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 实现名为 "foo" 的函数，使用 CompositeImplicitAutograd 模式，用于调试 fn_math
                lambda m: m.impl_t_t(
                    "foo", "CompositeImplicitAutograd", debug="fn_math"
                ),
                # 实现名为 "foo" 的函数，使用 FPGA 模式，用于调试 fn_fpga
                lambda m: m.impl_t_t("foo", "FPGA", debug="fn_fpga"),
            ],
        )
        # 提取结果状态和表格
        state, table = result.state, result.table
        # 断言状态与预期内联结果一致
        self.assertExpectedInline(
            state,
            """\
# 定义测试类 TestDispatchTable
class TestDispatchTable(TestCase):
    # 测试函数 test_computed_table_with_fpga
    def test_computed_table_with_fpga(self):
        # 调用 commute 函数，传入参数 "foo" 和 lambda 函数列表
        result = self.commute(
            "foo",
            [
                # m.def("foo(Tensor x) -> Tensor")
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # m.impl("foo", torch::kFPGA, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t("foo", "FPGA", debug="fn_fpga"),
                # m.impl("foo", torch::kCompositeImplicitAutograd, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t(
                    "foo", "CompositeImplicitAutograd", debug="fn_math"
                ),
            ],
        )
        # 获取结果的状态和表格
        state, table = result.state, result.table
        # 断言状态符合预期
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
FPGA: fn_fpga :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 提取感兴趣的键对应的调度表
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check + ("FPGA",)
        )

        # 断言提取的表格符合预期
        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: fn_math [math kernel]
CPU: fn_math [math kernel]
CUDA: fn_math [math kernel]
XLA: fn_math [math kernel]
AutogradOther: ambiguous_autogradother [ambiguous autogradother]
AutogradCPU: fn_math [math kernel]
AutogradCUDA: fn_math [math kernel]
AutogradXLA: fn_math [math kernel]
FPGA: fn_fpga [kernel]
""",
        )

    # 测试函数 test_computed_table_with_cpu_defaultbackend
    def test_computed_table_with_cpu_defaultbackend(self):
        # 调用 commute 函数，传入参数 "foo" 和 lambda 函数列表
        result = self.commute(
            "foo",
            [
                # m.def("foo(Tensor x) -> Tensor")
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
                # m.impl("foo", torch::kCompositeExplicitAutograd, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t(
                    "foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"
                ),
            ],
        )
        # 获取结果的状态和表格
        state, table = result.state, result.table
        # 断言状态符合预期
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
""",
        )

        # 提取感兴趣的键对应的调度表
        extracted_table = extract_dispatch_table_with_keys(
            table, dispatch_keys_to_check
        )

        # 断言提取的表格符合预期
        self.assertExpectedInline(
            extracted_table,
            """\
Undefined: fn_defaultbackend [default backend kernel]
CPU: fn_cpu [kernel]
CUDA: fn_defaultbackend [default backend kernel]
XLA: fn_defaultbackend [default backend kernel]
AutogradOther: registered in pytorch framework [backend fallback]
AutogradCPU: registered in pytorch framework [backend fallback]
AutogradCUDA: registered in pytorch framework [backend fallback]
AutogradXLA: registered in pytorch framework [backend fallback]
""",
        )
    # 定义测试方法：使用 CPU 自动求导默认后端测试计算表
    def test_computed_table_with_cpu_autograd_defaultbackend(self):
        # 调用 self.commute 方法执行测试，返回结果给 result
        result = self.commute(
            "foo",
            [
                # 注册 foo 函数的声明：m.def("foo(Tensor x) -> Tensor")
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 实现 foo 函数在 CPU 后端的版本：m.impl("foo", torch::kCPU, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
                # 实现 foo 函数在 Autograd 后端的版本：m.impl("foo", torch::kAutograd, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
                # 实现 foo 函数在 CompositeExplicitAutograd 后端的版本：m.impl("foo", torch::kCompositeExplicitAutograd, [](const Tensor & x) { return x })
                lambda m: m.impl_t_t(
                    "foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"
                ),
            ],
        )
        # 获取结果的状态和表格
        state, table = result.state, result.table
        # 断言状态和内联预期结果
        self.assertExpectedInline(
            state,
            """\
    def test_computed_table_with_cpu_autograd_math_defaultbackend(self):
        result = self.commute(
            "foo",
            [
                # 注册函数签名为 "foo(Tensor x) -> Tensor"
                lambda m: m.def_("foo(Tensor x) -> Tensor"),
                # 在 CPU 上实现函数 "foo"
                lambda m: m.impl_t_t("foo", "CPU", debug="fn_cpu"),
                # 在 Autograd 模式下实现函数 "foo"
                lambda m: m.impl_t_t("foo", "Autograd", debug="fn_autograd"),
                # 在 CompositeImplicitAutograd 模式下实现函数 "foo"
                lambda m: m.impl_t_t("foo", "CompositeImplicitAutograd", debug="fn_math"),
                # 在 CompositeExplicitAutograd 模式下实现函数 "foo"
                lambda m: m.impl_t_t("foo", "CompositeExplicitAutograd", debug="fn_defaultbackend"),
            ],
        )
        # 获取结果的状态和表格
        state, table = result.state, result.table
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: FROM_SCHEMA
CPU: fn_cpu :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
Autograd[alias]: fn_autograd :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeImplicitAutograd[alias]: fn_math :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
CompositeExplicitAutograd[alias]: fn_defaultbackend :: (Tensor _0) -> Tensor _0 [ boxed unboxed ]
"""
        )
    def test_def_with_explicit_alias(self):
        # 使用 self.commute 方法调用 "foo" 操作，并定义一个带有纯函数别名的新操作
        state = self.commute(
            "foo",
            [
                # 使用 lambda 函数定义一个新操作，注册名为 "foo(Tensor x, Tensor y) -> Tensor" 的函数
                lambda m: m.def_(
                    "foo(Tensor x, Tensor y) -> Tensor", alias="PURE_FUNCTION"
                )
            ],
        ).state
        # 断言操作的状态，检查注册的函数名称、架构和调试信息，并包括注册位置和别名分析类型
        self.assertExpectedInline(
            state,
            """\
name: test::foo
schema: test::foo(Tensor x, Tensor y) -> Tensor
debug: registered at /dev/null:0
alias analysis kind: PURE_FUNCTION
""",
        )

    def test_multiple_def_alias_defaulting(self):
        ops = [
            # 使用 lambda 函数定义一个新操作，注册名为 "foo(Tensor x) -> Tensor" 的函数，并指定纯函数别名
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # 使用 RegisterOperators().op 方法注册名为 "foo(Tensor x) -> Tensor" 的传统函数
            lambda m: m.def_legacy("foo(Tensor x) -> Tensor"),
        ]
        # 断言多次定义相同名称和重载的操作会引发异常，检查注册位置和重复注册信息
        self.assertExpectedInline(
            self.commute("foo", ops, expect_raises=True).state,
            """Tried to register an operator (test::foo(Tensor x) -> Tensor) with the same name and overload """
            """name multiple times. Each overload's schema should only be registered with a single call to def(). """
            """Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0""",
        )
    def test_multiple_def_alias_mismatch(self):
        ops = [
            # lambda 函数用于注册名为 "foo(Tensor x) -> Tensor" 的纯函数别名
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="PURE_FUNCTION"),
            # lambda 函数用于注册名为 "foo(Tensor x) -> Tensor" 的保守别名
            lambda m: m.def_("foo(Tensor x) -> Tensor", alias="CONSERVATIVE"),
        ]
        self.assertExpectedInline(
            # 调用 commute 方法，测试 "foo" 函数的多个注册操作
            self.commute("foo", ops, expect_raises=True).state,
            """Tried to register an operator (test::foo(Tensor x) -> Tensor) with the same name and overload """
            """name multiple times. Each overload's schema should only be registered with a single call to def(). """
            """Duplicate registration: registered at /dev/null:0. Original registration: registered at /dev/null:0""",
        )

    def test_multiple_fallback(self):
        # 获取全局分发对象
        global_m = C._dispatch_library("IMPL", "_", "XLA")
        # 添加后备处理过程
        global_m.fallback_fallthrough(),
        try:
            # 再次添加后备处理过程，预期抛出运行时错误
            global_m.fallback_fallthrough(),
        except RuntimeError as e:
            self.assertExpectedInline(
                str(e),
                """Tried to register multiple backend fallbacks for the same dispatch key XLA; previous registration """
                """registered at /dev/null:0, new registration registered at /dev/null:0""",
            )
        else:
            # 如果没有抛出错误，则断言失败
            self.assertTrue(False)

    def test_overwrite_math(self):
        ops = [
            # lambda 函数用于实现 "foo" 函数的第一个版本
            lambda m: m.impl_t_t("foo", debug="fn1"),
            # lambda 函数用于实现 "foo" 函数的第二个版本
            lambda m: m.impl_t_t("foo", debug="fn2"),
        ]
        # 断言测试：测试 "foo" 函数的重载顺序是否符合预期
        # 不是交换操作的（非交换的）情况
        self.assertExpectedInline(
            self.commute("foo", ops, ctor_order=(0, 1)).state,
            """\
# 创建一个测试类 TestC，继承自 unittest.TestCase，用于测试某个模块的功能
class TestC(TestCase):

    # test_find_dangling_impls 方法用于测试寻找悬空实现的功能
    def test_find_dangling_impls(self):
        # 调用 C 模块的 _dispatch_find_dangling_impls 方法，获取所有悬空实现
        dangling_impls = C._dispatch_find_dangling_impls()
        # 断言悬空实现的数量为 0
        self.assertEqual(
            0,
            len(dangling_impls),
            msg=f"Expect zero dangling impls, but found: {dangling_impls}",
        )

    # test_find_dangling_impls_ext 方法用于测试通过扩展寻找悬空实现的功能
    def test_find_dangling_impls_ext(self):
        # 获取 dangling_impl_extension.cpp 文件的绝对路径
        extension_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "cpp_extensions",
            "dangling_impl_extension.cpp",
        )
        # 加载名为 dangling_impl_extension 的 C++ 扩展模块
        module = torch.utils.cpp_extension.load(
            name="dangling_impl_extension",
            sources=[
                extension_path,
            ],
            extra_cflags=["-g"],
            verbose=True,
        )
        # 调用 C 模块的 _dispatch_find_dangling_impls 方法，获取所有悬空实现
        impls = C._dispatch_find_dangling_impls()
        # 断言悬空实现的数量为 1
        self.assertEqual(1, len(impls))
        # 断言悬空实现的具体信息与预期一致
        self.assertEqual(
            f"""\
name: __test::foo
schema: (none)
CPU: registered at {extension_path}:5 :: () -> () [ boxed unboxed ]
""",
            impls[0],
        )

    # test_dispatch_print_registrations_for_dispatch_key_invalid 方法用于测试对无效 dispatch key 的异常处理
    def test_dispatch_print_registrations_for_dispatch_key_invalid(self):
        # 断言调用 _dispatch_print_registrations_for_dispatch_key 方法时会抛出 RuntimeError 异常，异常信息包含 "could not parse dispatch key: invalid_key"
        with self.assertRaisesRegex(
            RuntimeError, "could not parse dispatch key: invalid_key"
        ):
            C._dispatch_print_registrations_for_dispatch_key("invalid_key")


# TestPythonDispatcher 类用于测试 PythonDispatcher 类的功能
class TestPythonDispatcher(TestCase):

    # test_basic 方法用于测试基本的注册和调度功能
    def test_basic(self):
        # 创建 PythonDispatcher 实例
        dispatcher = PythonDispatcher()
        # 注册多个 dispatch key
        dispatcher.register(["CPU", "XLA", "Lazy", "CompositeImplicitAutograd"])
        # 断言调用 dispatchTable 方法返回的计算后的调度表内容符合预期
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            """\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
Lazy            fn_Lazy [kernel]
FPGA            fn_CompositeImplicitAutograd [math kernel]
AutogradOther   fn_CompositeImplicitAutograd [math kernel]
AutogradCPU     [backend fallback]
AutogradXLA     [backend fallback]
AutogradLazy    [backend fallback]
""",
        )

    # test_math_autogradcpu 方法用于测试包含 AutogradCPU 的注册和调度功能
    def test_math_autogradcpu(self):
        # 创建 PythonDispatcher 实例
        dispatcher = PythonDispatcher()
        # 注册多个 dispatch key，包括 AutogradCPU
        dispatcher.register(
            ["CPU", "XLA", "Lazy", "CompositeImplicitAutograd", "AutogradCPU"]
        )
        # 断言调用 dispatchTable 方法返回的计算后的调度表内容符合预期
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            """\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
Lazy            fn_Lazy [kernel]
FPGA            fn_CompositeImplicitAutograd [math kernel]
""",
        )
    def test_defaultbackend(self):
        # 创建 PythonDispatcher 的实例
        dispatcher = PythonDispatcher()
        # 注册多个核心类型到调度器中
        dispatcher.register(
            ["CPU", "XLA", "Lazy", "CompositeExplicitAutograd", "AutogradCPU"]
        )
        # 断言调度表的期望输出
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            """\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_XLA [kernel]
Lazy            fn_Lazy [kernel]
FPGA            fn_CompositeExplicitAutograd [default backend kernel]
AutogradOther   [backend fallback]
AutogradCPU     fn_AutogradCPU [kernel]
AutogradXLA     [backend fallback]
AutogradLazy    [backend fallback]
""",
        )

        # 断言注册表的期望输出
        self.assertExpectedInline(
            dispatcher.registrations(),
            """\

Registered Kernels
key             kernel
---------------------------
CPU             fn_CPU
XLA             fn_XLA
Lazy            fn_Lazy
AutogradCPU     fn_AutogradCPU
CompositeExplicitAutograd[alias] fn_CompositeExplicitAutograd
""",
        )

    def test_autogradother(self):
        # 创建 PythonDispatcher 的实例
        dispatcher = PythonDispatcher()
        # 注册多个核心类型到调度器中
        dispatcher.register(["CPU", "FPGA", "CompositeImplicitAutograd"])
        # 断言调度表的期望输出
        self.assertExpectedInline(
            dispatcher.dispatchTable(),
            """\

Computed Dispatch Table
key             kernel
---------------------------
CPU             fn_CPU [kernel]
XLA             fn_CompositeImplicitAutograd [math kernel]
Lazy            fn_CompositeImplicitAutograd [math kernel]
FPGA            fn_FPGA [kernel]
AutogradOther   ambiguous_autogradother [ambiguous autogradother]
AutogradCPU     [backend fallback]
AutogradXLA     fn_CompositeImplicitAutograd [math kernel]
AutogradLazy    fn_CompositeImplicitAutograd [math kernel]
""",
        )

        # 断言注册表的期望输出
        self.assertExpectedInline(
            dispatcher.registrations(),
            """\

Registered Kernels
key             kernel
---------------------------
FPGA            fn_FPGA
CPU             fn_CPU
CompositeImplicitAutograd[alias] fn_CompositeImplicitAutograd
""",
        )

    def test_duplicate_registrations(self):
        # 创建 PythonDispatcher 的实例
        dispatcher = PythonDispatcher()

        # 检查重复注册时是否引发 RuntimeError
        with self.assertRaisesRegex(RuntimeError, r"Overriden is not allowed"):
            dispatcher.register(["CPU", "CPU"])
    # 定义一个测试方法，用于测试默认后端数学运算
    def test_defaultbackend_math(self):
        # 创建一个 Python 调度器对象
        dispatcher = PythonDispatcher()

        # 使用断言检查是否抛出预期的 RuntimeError 异常，
        # 异常信息指出同时注册 CompositeImplicitAutograd 和 CompositeExplicitAutograd 是不允许的
        with self.assertRaisesRegex(
            RuntimeError,
            r"Registration to both CompositeImplicitAutograd and CompositeExplicitAutograd is not allowed",
        ):
            # 尝试注册两个后端类型 ["CompositeExplicitAutograd", "CompositeImplicitAutograd"]
            dispatcher.register(
                ["CompositeExplicitAutograd", "CompositeImplicitAutograd"]
            )

    # 定义一个测试方法，用于测试未实现的量化结构
    def test_quantized_structured_not_implemented(self):
        # 创建两个张量 x 和 y，形状为 [1, 1, 1]，元素值都为 0
        x = torch.zeros([1, 1, 1])
        y = torch.zeros([1, 1, 1])
        scale, zero_point = 1.0, 0
        dtype = torch.qint8
        # 对张量 x 和 y 进行量化，使用给定的比例尺度、零点和数据类型
        qx = torch.quantize_per_tensor(x, scale, zero_point, dtype)
        qy = torch.quantize_per_tensor(y, scale, zero_point, dtype)
        
        # 使用断言检查是否抛出预期的 NotImplementedError 异常，
        # 异常信息指出在 QuantizedCPU 后端上不能执行 'aten::bmm.out' 操作
        self.assertRaisesRegex(
            NotImplementedError,
            "Could not run 'aten::bmm.out' with arguments from the 'QuantizedCPU' backend.",
            # 使用 lambda 函数调用 torch.bmm(qx, qy)，捕获其抛出的异常
            lambda: torch.bmm(qx, qy),
        )
if __name__ == "__main__":
    # 检查当前模块是否作为主程序运行
    run_tests()
```