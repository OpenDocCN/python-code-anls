# `.\pytorch\test\inductor\test_custom_post_grad_passes.py`

```
# Owner(s): ["module: inductor"]
# 导入上下文管理工具、运算符操作及默认字典数据结构
import contextlib
import operator
from collections import defaultdict

# 导入 PyTorch 相关模块
import torch
import torch._inductor.pattern_matcher as pattern_matcher
import torch.fx as fx
from torch._dynamo.utils import counters

# 导入 PyTorch _inductor 模块的配置及降低操作
from torch._inductor import config
from torch._inductor.lowering import lowerings as L
from torch._inductor.pattern_matcher import Arg, CallFunction, PatternMatcherPass

# 导入测试相关模块：运行测试及测试用例
from torch._inductor.test_case import run_tests, TestCase

# 导入内部测试工具函数：判断是否为 Linux 系统
from torch.testing._internal.common_utils import IS_LINUX

# 导入内部 _inductor_utils 模块：判断是否具有 CPU
from torch.testing._internal.inductor_utils import HAS_CPU


@config.patch({"freezing": True})
class TestCustomPassBase(TestCase):
    def _clone_inputs(self, inputs):
        # 复制输入参数的函数，对于非 Tensor 类型的输入，直接返回，否则进行克隆操作
        def clone(x):
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()

        return tuple(clone(x) for x in inputs)

    def _test_common(
        self,
        mod,
        inputs,
        matcher_count,
        matcher_nodes,
        atol=1e-5,
        rtol=1.3e-6,
    ):
        # 清除计数器
        counters.clear()
        
        # 可能的自动转换上下文管理
        maybe_autocast = contextlib.nullcontext()
        
        # 使用 torch.no_grad() 和 maybe_autocast 上下文环境，执行以下操作
        with torch.no_grad(), maybe_autocast:
            # 克隆输入参数
            clone_inputs = self._clone_inputs(inputs)
            
            # 计算预期输出
            expected = mod(*inputs)
            
            # 编译模型并获取实际输出
            actual = torch.compile(mod)(*clone_inputs)
            
            # 使用指定的误差容差检查实际输出和预期输出的一致性
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
            
            # 断言统计计数器中的模式匹配器数量
            self.assertEqual(
                counters["inductor"]["pattern_matcher_count"], matcher_count
            )
            
            # 断言统计计数器中的模式匹配器节点数量
            self.assertEqual(
                counters["inductor"]["pattern_matcher_nodes"],
                matcher_nodes,
            )


aten = torch.ops.aten
mkldnn = torch.ops.mkldnn


def change_cos_pass(graph):
    # 遍历图中的节点，将所有调用 torch.ops.aten.cos.default 的节点修改为调用 torch.ops.aten.sin.default
    for node in graph.nodes:
        if node.op == "call_function" and node.target == aten.cos.default:
            node.target = aten.sin.default


class TestPostGradCustomPrePostPass(TestCustomPassBase):
    # MKLDNN 融合的模式匹配器
    # （位于 torch/_inductor/fx_passes/mkldnn_fusion.py），
    # 并将其应用于自定义的 post_grad_passes。
    # 注册 MKLDNN 卷积ReLU融合的模式
    def _register_mkldnn_conv_relu_fusion(self, custom_pass_dict):
        # 定义 MKLDNN 卷积ReLU融合的模式
        def _mkldnn_conv_relu_pattern():
            return CallFunction(
                aten.relu,  # 调用 PyTorch 的ReLU函数
                CallFunction(
                    mkldnn._convolution_pointwise.default,  # 调用MKLDNN的默认卷积函数
                    Arg(),  # 卷积函数的第一个参数
                    Arg(),  # 卷积函数的第二个参数
                    Arg(),  # 卷积函数的第三个参数
                    Arg(),  # 卷积函数的第四个参数
                    Arg(),  # 卷积函数的第五个参数
                    Arg(),  # 卷积函数的第六个参数
                    Arg(),  # 卷积函数的第七个参数
                    Arg(),  # 卷积函数的第八个参数
                    Arg(),  # 卷积函数的第九个参数
                    Arg(),  # 卷积函数的第十个参数
                    _users=1,  # 卷积函数的用户数设为1
                ),
            )

        # 注册融合降低的模式匹配工具函数
        def _register_fusion_lowering(pattern, custom_pass_dict):
            # 设置一个虚拟检查函数，始终返回True
            def dummy_check(m):
                return True

            # 注册自定义降低模式函数
            def register_custom_lowering_pattern(
                pattern, extra_check, custom_pass_dict
            ):
                return pattern_matcher.register_lowering_pattern(
                    pattern, extra_check, pass_dict=custom_pass_dict
                )

            # 使用注册自定义降低模式函数注册特定模式的融合降低
            @register_custom_lowering_pattern(pattern, dummy_check, custom_pass_dict)
            def fn(match, *args, **kwargs):
                # 构造计算参数列表，用于后续的MKLDNN卷积点对点默认函数调用
                computation_args = list(args)[:-3] + ["relu", [], ""]
                return L[mkldnn._convolution_pointwise.default](*computation_args)

            return fn

        # 调用注册融合降低的模式匹配工具函数，并传入MKLDNN卷积ReLU融合的模式和自定义的传递字典
        _register_fusion_lowering(_mkldnn_conv_relu_pattern(), custom_pass_dict)

    # 自定义后传递的案例模型
    class _CustomPass(PatternMatcherPass):
        def __init__(self):
            super().__init__()

        # 定义用于图形g的调用函数
        def __call__(self, g: torch.fx.graph.Graph):
            self.apply(g)

    # 案例模型
    class _ConvReLU(torch.nn.Module):
        def __init__(self, ic, oc):
            super().__init__()
            # 定义卷积层和ReLU层
            self.conv = torch.nn.Conv2d(ic, oc, kernel_size=3, stride=1, padding=1)

        # 定义前向传播函数
        def forward(self, x):
            x1 = self.conv(x)
            return x1.relu()

    # 测试自定义联合传递的预处理
    def test_custom_joint_pass_pre(self):
        # 使用config.patch装饰器，并设置联合自定义预传递
        with config.patch(joint_custom_pre_pass=change_cos_pass):

            # 定义函数g，对输入x进行多次sin函数调用
            def g(x):
                return x.sin().sin().sin()

            # 定义函数f，对输入x进行多次cos函数调用
            def f(x):
                return x.cos().cos().cos()

            # 生成随机张量x，并使用torch.compile对函数f进行编译和运行，并使用torch.testing.assert_close断言结果
            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))

    # 测试自定义联合传递的后处理
    def test_custom_joint_pass_post(self):
        # 使用config.patch装饰器，并设置联合自定义后传递
        with config.patch(joint_custom_post_pass=change_cos_pass):

            # 定义函数g，对输入x进行多次sin函数调用
            def g(x):
                return x.sin().sin().sin()

            # 定义函数f，对输入x进行多次cos函数调用
            def f(x):
                return x.cos().cos().cos()

            # 生成随机张量x，并使用torch.compile对函数f进行编译和运行，并使用torch.testing.assert_close断言结果
            x = torch.randn(8, dtype=torch.float32)
            torch.testing.assert_close(torch.compile(f)(x), g(x))
    def test_custom_pre_pass(self):
        # 使用 config.patch 上下文管理器设置测试环境
        with config.patch(
            # 禁用模式匹配器，仅在 post_grad_passes() 中留下自定义预处理
            pattern_matcher=False,
            # 设置 post_grad_custom_pre_pass 为自定义的 _CustomPass 实例
            post_grad_custom_pre_pass=self._CustomPass(),
            # 将 pattern match 定义为自定义的后处理优化 pass
            post_grad_custom_post_pass=None,
        ):
            # 在 config.post_grad_custom_pre_pass 上注册 mkldnn conv relu 融合
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_pre_pass)

            # 创建一个评估模式下的 _ConvReLU 实例
            mod = self._ConvReLU(16, 16).eval()
            # 创建输入张量 x，形状为 (1, 16, 56, 56)，数据类型为 torch.float32
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            # 设置匹配计数和节点数
            match_count = 1
            match_nodes = 2
            # 设置其他匹配计数和节点数，用于 conv 预打包权重
            other_match_count = 1
            other_match_nodes = 1
            # 运行通用测试函数 _test_common
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,  # 期望的匹配计数
                match_nodes + other_match_nodes,  # 期望的节点数
            )

    def test_custom_post_pass(self):
        # 使用 config.patch 上下文管理器设置测试环境
        with config.patch(
            # 禁用模式匹配器，仅在 post_grad_passes() 中留下自定义后处理
            pattern_matcher=False,
            # 将 post_grad_custom_pre_pass 设置为 None，表示没有自定义预处理
            post_grad_custom_pre_pass=None,
            # 设置 post_grad_custom_post_pass 为自定义的 _CustomPass 实例
            post_grad_custom_post_pass=self._CustomPass(),
        ):
            # 在 config.post_grad_custom_post_pass 上注册 mkldnn conv relu 融合
            self._register_mkldnn_conv_relu_fusion(config.post_grad_custom_post_pass)

            # 创建一个评估模式下的 _ConvReLU 实例
            mod = self._ConvReLU(16, 16).eval()
            # 创建输入张量 x，形状为 (1, 16, 56, 56)，数据类型为 torch.float32
            x = torch.randn((1, 16, 56, 56), dtype=torch.float32)

            # 设置匹配计数和节点数
            match_count = 1
            match_nodes = 2
            # 设置其他匹配计数和节点数，用于 conv 预打包权重
            other_match_count = 1
            other_match_nodes = 1
            # 运行通用测试函数 _test_common
            self._test_common(
                mod,
                (x,),
                match_count + other_match_count,  # 期望的匹配计数
                match_nodes + other_match_nodes,  # 期望的节点数
            )
    # 定义测试方法，用于测试自定义的前向传播梯度处理
    def test_custom_pre_grad_pass(self):
        # 保存图形对象的列表，初始为空
        saved_graph = [None]

        # 定义合并具有共享右手边的矩阵乘法的函数
        def merge_mm_shared_rhs(graph: fx.Graph):
            """
            Bad POC of merging mm with a shared RHS.
            i.e. [mm(x, W), mm(x2, W)] => mm(cat(x, x2), W).split()

            Isn't actually safe for a couple reasons. For example, it doesn't handle the
            case where the LHS inputs depend on each other
            """
            # 将传入的图形对象保存到 saved_graph 列表的第一个位置
            saved_graph[0] = graph
            # 找到所有矩阵乘法操作节点
            matmuls = [n for n in graph.nodes if n.target == torch.mm]
            # 使用 defaultdict 创建一个集合字典，用于存储右手边的值和相关的乘法操作节点
            rhs_vals = defaultdict(set)
            for m in matmuls:
                rhs_vals[m.args[1]].add(m)

            # 为图中的每个节点分配一个顺序索引
            order = {}
            for idx, n in enumerate(graph.nodes):
                order[n] = idx

            # 遍历右手边值和相关的乘法操作节点
            for rhs, matmuls in rhs_vals.items():
                # 如果只有一个乘法操作节点，跳过
                if len(matmuls) == 1:
                    continue
                # 对乘法操作节点按照它们在图中的顺序进行排序
                matmuls = sorted(matmuls, key=lambda x: order[x])
                # 在第一个乘法操作节点之前插入新节点
                with graph.inserting_before(matmuls[0]):
                    # 获取左手边的值列表
                    lhs_vals = [m.args[0] for m in matmuls]
                    # 创建一个新的 torch.cat 函数调用节点，将左手边值连接起来
                    new_cat = graph.create_node(
                        "call_function", torch.cat, args=(lhs_vals, 0)
                    )
                    # 创建一个新的 torch.mm 函数调用节点，使用新的连接值和右手边值
                    new_mm = graph.create_node(
                        "call_function", torch.mm, args=(new_cat, rhs)
                    )
                    # 创建一个新的 torch.split 函数调用节点，对新的乘法结果进行分割
                    split_vals = graph.create_node(
                        "call_function",
                        torch.split,
                        args=(
                            new_mm,
                            [l.meta["example_value"].shape[0] for l in lhs_vals],
                        ),
                    )
                # 更新每个乘法操作节点，将它们的目标设置为 operator.getitem，并设置新的参数
                for idx, m in enumerate(matmuls):
                    m.target = operator.getitem
                    m.args = (split_vals, idx)

        # 使用 @config.patch 装饰器，将自定义的前向传播梯度处理函数应用到 inner_test 函数
        @config.patch(pre_grad_custom_pass=merge_mm_shared_rhs)
        def inner_test():
            # 定义一个编译后的 torch 函数 f，接受权重 W 和嵌套序列作为参数
            @torch.compile
            def f(W, nested_seqs):
                # 对嵌套序列中的每个张量 s，计算 torch.mm(s, W)，将结果存储在 outs 列表中
                outs = [torch.mm(s, W) for s in nested_seqs]
                return outs

            # 随机生成一个 16x16 的浮点数权重张量 W
            W = torch.randn(16, 16, dtype=torch.bfloat16)
            # 创建嵌套序列，每个子序列长度分别为 4, 8, 5, 3，每个张量形状为 (l, 16)
            nested_seqs = [
                torch.randn(l, 16, dtype=torch.bfloat16) for l in [4, 8, 5, 3]
            ]

            # 调用 f 函数，执行前向传播
            f(W, nested_seqs)
            # 断言 saved_graph[0] 不为空，确保图形对象已经被保存
            assert saved_graph[0] is not None
            # 获取所有图形中目标为 torch.mm 的节点列表
            matmuls = [n for n in saved_graph[0].nodes if n.target == torch.mm]
            # 断言 torch.mm 节点的数量为 1，确保合并操作成功完成
            assert len(matmuls) == 1

        # 执行 inner_test 函数，测试自定义前向传播梯度处理的功能
        inner_test()
# 如果脚本作为主程序运行
if __name__ == "__main__":
    # 如果运行环境是 Linux，并且有可用的 CPU，且支持 MKLDNN 加速
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        # 运行测试函数
        run_tests()
```